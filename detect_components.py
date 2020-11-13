# AMME4710 - EJH Detector 2020
# Circuit Digitaliser 

# 470205127
# 470355499
# 470425954

import cv2
import numpy as np
import time

from helper import *

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
    img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_DEFAULT)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Preprocess the image to extract components
    binarised = get_mask(gray, pad)

    # Dilation to connect the circuit up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,30))
    # closing = cv2.dilate(closing, kernel)
    
    # Canny edge detection 
    # canny = cv2.Canny(binarised, 100, 200)
    cv2.imshow('Og mask', binarised)
    # cv2.waitKey(0)
    # canny_dil = cv2.dilate(canny, kernel)   # Dilate to join them up 
    canny_dil = cv2.morphologyEx(binarised, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imshow('Og mask', canny_dil)
    cv2.waitKey(0)

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
    cv2.imshow('bin cropped', cropped)
    # cv2.waitKey(0)
    
    _, cropped = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imshow('cropped', cropped)

    ##### Perform component extraction on this cropped image #####
    # Gaussian Blur
    # gray_crop = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    # thr_crop = get_mask(gray_crop)
    cv2.imshow('bin cropped', cropped)

    big_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (35,35))

    # Closing operation to turn components into blobs
    mask = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, big_kern)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0) 
    # view_contours(mask)
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
        # print(len(contours))
        conts.append(len(contours))
        if max_cont < len(contours): 
            if len(contours) - max_cont > 2:
                peaked = True
                # print("Peaked!")
            max_cont = len(contours)
        elif (max_cont+init_cont)/2 >= len(contours) and peaked:
            print("Peak found at kernel size", i)
            found = True
            break
        # cv2.imshow('mask', mask2)
        # cv2.waitKey(0) 

    # Not found when the circuit is dense
    if not found:
        line_width = (conts.index(max(conts)) ) *2
        print('Peak asssumed to be at', line_width)
    else:
        line_width = i*2
    print(line_width)
    # Set a cap on the linewidth
    line_width = max(20, min(30, line_width))
    print(line_width)
    # Do it all over again, moderated by the linewidth
    big_line_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20+line_width,20+line_width))
    line_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_width,line_width))

    # Closing operation to turn components into blobs
    mask_line = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, big_line_kern)
    cv2.imshow('mask_line', mask_line)
    # cv2.waitKey(0)

    mask_line = fill_small_contours(mask_line)
    cv2.imshow('mask_line', mask_line)
    # cv2.waitKey(0)

    # Get rid of the lines
    mask_no_line = cv2.morphologyEx(mask_line, cv2.MORPH_OPEN, line_kern)
    cv2.imshow('mask_line', mask_no_line)
    # cv2.waitKey(0)

    small_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask_no_line = cv2.morphologyEx(mask_no_line, cv2.MORPH_CLOSE, small_kern)
    
    # # Remove outlier blobs generated from line intersections
    mask_no_line = reject_outliers(mask_no_line)

    # # Med dilation to get the component shapes back up to size
    # mask_no_line = cv2.dilate(mask_no_line, line_kern, iterations=1)
    # cv2.imshow('mask_no_line', mask_no_line)
    # cv2.waitKey(0)

    # Get the bounding boxes for all the remaining components
    bboxes = np.array([0,0,0,0])
    contours, _ = cv2.findContours(mask_no_line, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        # Iterate here to potentially break down the contour into small ones
        sub_boxes = decompose_contour(mask_no_line, [x,y,w,h], line_width)

        # Shift it back to account for the padding
        # x = x-pad
        # y = y-pad

        # Reject bboxes based on aspect ratio
        if 4 > w/h > 0.25:  
            # Blow up the contour a bit to make sure all components are fully captured
            cv2.rectangle(markup, (int(x-0.1*w), int(y-0.1*h)), (int(x+1.1*w), int(y+1.1*h)), (0,255,0), 2)
            cv2.putText(markup, str(cv2.contourArea(contour)), (x,y+h), cv2.FONT_HERSHEY_COMPLEX, 1, (0,70,255))

            # Store the bbox
            bboxes = np.vstack( (bboxes, np.array([x,y,w,h])) )

    cv2.imshow('cropped', markup)
    print('Press any key to continue...')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return bboxes[1:], cropped_img, mask_line

if __name__ == '__main__':
    # Import images 
    for i in range(8,15):
        path = './Data/Circuits/' + str(i) + '.jpg'
        print(i)
        img = cv2.imread(path)
        bboxes = detect_components(img)
    print(bboxes)
    # names = ['5','10','15','15manny','20','25','35','40','40many','60','70','70_2','80','80_2','80_3','90','90_2']
    # for i in range(1, len(names)):
    #     path = './scale/' + names[i] + '.jpg'
    #     img = cv2.imread(path)
    #     bboxes = detect_components(img)
