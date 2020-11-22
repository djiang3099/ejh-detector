# AMME4710 - EJH Detector 2020
# Circuit Digitaliser 

# 470205127
# 470355499
# 470425954

import cv2
import numpy as np
import time

from helper import *

# This function takes in a CV2 image input and performs a series of morphological operations to 
# identify component locations as bounding boxes. 

# Note that cv2.imshow() and related lines have been commented in compliance with Google Colab
def detect_components(img):
    # Resize the image to fit on the screen
    height, width = img.shape[:2]
    lim_dim = 550
    pad = 40

    # Find which length is longer and scale it appropriately 
    if height > width:
        scale = width/lim_dim
    else:
        scale = height/lim_dim
    img = cv2.resize(img, ( int(width/scale), int(height/scale) ), dst=img, interpolation = cv2.INTER_CUBIC)
    
    # Add padding to enable coherent cropping later
    img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_DEFAULT)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Preprocess the image to extract components
    binarised = get_mask(gray, pad)
    # cv2.imshow('Og mask', binarised)

    # Dilation to connect the circuit up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,30))
    canny_dil = cv2.morphologyEx(binarised, cv2.MORPH_CLOSE, kernel, iterations=1)
    # cv2.imshow('Og mask', canny_dil)

    # Crop the circuit out of the image and rotate to straighten
    cropped = extract_circuit(binarised, canny_dil, crop_margin=0.1)
    cropped = cv2.copyMakeBorder(cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0,0,0])
    
    markup = extract_circuit(img_pad, canny_dil, crop_margin=0.1)
    markup = cv2.copyMakeBorder(markup, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    # Rescale the cropped image to normalise
    roi_height, roi_width = cropped.shape[:2]
    if roi_height > roi_width:
        scale = roi_width/lim_dim
    else:
        scale = roi_height/lim_dim

    cropped = cv2.resize(cropped, ( int(roi_width/scale), int(roi_height/scale) ), interpolation = cv2.INTER_CUBIC)
    markup = cv2.resize(markup, ( int(roi_width/scale), int(roi_height/scale) ), interpolation = cv2.INTER_CUBIC)
    cropped_img = markup.copy()
    # cv2.imshow('Binary cropped', cropped)
    
    _, cropped = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    ##### Perform component extraction on this cropped image #####
    # cv2.imshow('bin cropped', cropped)

    big_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (35,35))

    # Closing operation to turn components into blobs
    mask = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, big_kern)
    mask = fill_small_contours(mask)

    # Erosion to get rid of lines
    init_cont = len(cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2])
    max_cont = init_cont
    peaked = False
    found = False
    conts = []

    # Slowly erode the lines with increasing kernel size to determine linewidth, when 
    # components begin forming separate contours
    for i in range(3,25):
        mid_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i,i))
        mask2 = cv2.erode(mask, mid_kern, iterations=1)
        contours, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        conts.append(len(contours))
        if max_cont < len(contours): 
            if len(contours) - max_cont > 2:
                peaked = True
            max_cont = len(contours)
        elif (max_cont+init_cont)/2 >= len(contours) and peaked:
            print("Peak found at kernel size", i)
            found = True
            break
        cv2.putText(mask2, str(i*2), (50,80), cv2.FONT_HERSHEY_COMPLEX, 1, 255)

    # Not found when the circuit is dense
    if not found:
        line_width = (conts.index(max(conts)) ) *2
        print('Peak asssumed to be at', line_width)
    else:
        line_width = i*2

    # Set a cap on the linewidth
    line_width = max(20, min(30, line_width))
    print('Line width set to:', line_width)

    # Do it all over again, moderated by the linewidth
    big_line_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20+line_width,20+line_width))
    line_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_width,line_width))

    # Closing operation to turn components into blobs
    mask_line = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, big_line_kern)
    # cv2.imshow('mask_line', mask_line)

    mask_line = fill_small_contours(mask_line)
    # cv2.imshow('mask_line', mask_line)

    # Get rid of the lines
    mask_no_line = cv2.morphologyEx(mask_line, cv2.MORPH_OPEN, line_kern)
    # cv2.imshow('mask_line', mask_no_line)

    small_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask_no_line = cv2.morphologyEx(mask_no_line, cv2.MORPH_CLOSE, small_kern)
    
    # Find all the contours now, and convert to bboxes
    bboxes = np.array([0,0,0,0])
    contours, _ = cv2.findContours(mask_no_line, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)

        # Iterate here to potentially break down the contour into small ones
        sub_boxes = decompose_contour(cropped, [x,y,w,h], line_width)
        if sub_boxes.ndim > 1:
            bboxes = np.vstack( (bboxes, sub_boxes[1:]) )

    # Remove outlier blobs generated from line intersections
    filt_bboxes = reject_area_outliers(bboxes[1:])
    for x,y,w,h in filt_bboxes:
        cv2.rectangle(markup, (int(x-0.15*w), int(y-0.15*h)), (int(x+1.15*w), int(y+1.15*h)), (0,255,0), 2)

    cv2.imshow('cropped', markup)
    print('Press any key to continue...')
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return filt_bboxes, cropped_img, mask_line

if __name__ == '__main__':
    # Import images 
    for i in range(1,3):
        path = './Data/Circuits/' + str(i) + '.jpg'
        print('Image number', i)
        img = cv2.imread(path) 
        bboxes = detect_components(img)