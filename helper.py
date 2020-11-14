# AMME4710 - EJH Detector 2020
# Circuit Digitaliser 

# 470205127
# 470355499
# 470425954

import cv2
import numpy as np

def enlarge_rotated_rect(box, scale=0.05):
    pt1 = box[0]
    pt2 = box[1]
    pt3 = box[2]
    pt4 = box[3]
    big_box = np.zeros((4,2))
    big_box[0] = [pt1[0] + (pt1[0]-pt3[0])*scale/2, pt1[1] + (pt1[1]-pt3[1])*scale/2]
    big_box[2] = [pt3[0] - (pt1[0]-pt3[0])*scale/2, pt3[1] - (pt1[1]-pt3[1])*scale/2]
    big_box[1] = [pt2[0] + (pt2[0]-pt4[0])*scale/2, pt2[1] + (pt2[1]-pt4[1])*scale/2]
    big_box[3] = [pt4[0] - (pt2[0]-pt4[0])*scale/2, pt4[1] - (pt2[1]-pt4[1])*scale/2]
    return np.int0(big_box)

def crop_n_rotate(img, box, size):
    width = size[0]
    height = size[1]
    src_pts = box.astype("float32")
    # End coordinates is just same as width/height
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # Get perspective transform from src to dst points and warp
    transform = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, transform, size)
    return warped

def remove_noise(mask):
    reject_mask = np.zeros(mask.shape[:2], dtype="uint8")
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    areas = np.array([])
    for contour in contours: 
        areas = np.append(areas, cv2.contourArea(contour))

    avg = np.mean(areas)
    idx = np.where(areas < 300)[0]
    reject_cnt = np.array(contours)[idx]
    cv2.drawContours(reject_mask, reject_cnt, -1, 255, -1)
    reject_mask = cv2.bitwise_not(reject_mask)

    return cv2.bitwise_and(mask, reject_mask)

def get_mask(img, pad=20):
    # Gaussian Blur
    blur = cv2.GaussianBlur(img, (5,5), 0)
    blur = cv2.GaussianBlur(blur, (5,5), 0)

    # Adaptive thresholding to binarise the image
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY_INV, 11, 2)
    
    th3 = cv2.copyMakeBorder(th3, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0,0,0])
    
    # Closing operation to fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel, iterations=2)

    closing = cv2.erode(closing, kernel, iterations=1)
    cv2.imshow('thr', closing)
    # cv2.waitKey(0)
    
    closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Get rid of noise (small contours)
    closing = remove_noise(closing)
    cv2.imshow('thr', closing)
    # cv2.waitKey(0)

    # Closing again to connect up the circuit
    # closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closing

def extract_circuit(img, mask, crop_margin=0.1):
    # Identify ROI for circuit and crop it out, TODO: Use 4sided polygon
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    best_contour = None
    for contour in contours: 
        if best_contour is None:
            best_contour = contour
        elif cv2.contourArea(contour) > cv2.contourArea(best_contour):
            best_contour = contour

    # Get the min enclosing rectangle, should contain the full circuit
    rect = cv2.minAreaRect(best_contour)
    box = np.int0(cv2.boxPoints(rect))  # Box points are clockwise starting from the lowest one
    big_box = enlarge_rotated_rect(box, crop_margin)

    # Extract this ROI and straighten it
    roi_width = int((1+crop_margin)*rect[1][0])
    roi_height = int((1+crop_margin)*rect[1][1])
    cropped = crop_n_rotate(img, big_box, (roi_width, roi_height))
    return cropped

def reject_outliers(mask):
    valid_mask = np.zeros(mask.shape[:2], dtype="uint8")
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    areas = np.array([])
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour)
        areas = np.append(areas, w*h)

    # Get rid of random noise contours
    idx = np.where(areas > 300)[0]
    filt_areas = areas[idx]

    # Set a threshold based off mean contour size
    try:
        if np.average(filt_areas) / np.std(filt_areas) > 2:
            print('Similarly sized contours', np.average(filt_areas) / np.std(filt_areas))
            # There are few outliers, reject two std dev away from avg
            idx = np.where(areas > 0.5*np.average(filt_areas))[0]
        else: 
            print('Significant outliers', np.average(filt_areas) / np.std(filt_areas))
            # Mean is similar to std dev, meaning significant number of outliers
            idx = np.where(areas > 0.25*np.average(filt_areas))[0]
    except: 
        print('No contours found for this mask')
        return mask
    # Keep the contours specified by idx
    filt_cnt = np.array(contours)[idx]

    for contour in filt_cnt:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(valid_mask, (x,y), (x+w, y+h), 255, -1)

    return cv2.bitwise_and(valid_mask, mask), filt_cnt


def reject_area_outliers(bboxes):
    areas = np.array([])
    for _,_,w,h in bboxes: 
        areas = np.append(areas, w*h)

    # Get rid of random noise contours
    idx = np.where(areas > 300)[0]
    filt_areas = areas[idx]

    # Set a threshold based off mean contour size
    try:
        # if np.average(filt_areas) / np.std(filt_areas) > 2:
        #     print('Similarly sized contours', np.average(filt_areas) / np.std(filt_areas))
        #     # There are few outliers, reject two std dev away from avg
        #     idx = np.where(areas > 0.5*np.average(filt_areas))[0]
        # else: 
        #     print('Significant outliers', np.average(filt_areas) / np.std(filt_areas))
        #     # Mean is similar to std dev, meaning significant number of outliers
        #     idx = np.where(areas > 0.25*np.average(filt_areas))[0]
        idx = np.where(areas > 0.5*np.average(filt_areas))[0]
    except: 
        print('No contours found for this mask')
        return bboxes
    # Keep the contours specified by idx
    filt_bboxes = bboxes[idx]

    # for contour in filt_cnt:
    #     x,y,w,h = cv2.boundingRect(contour)
    #     cv2.rectangle(valid_mask, (x,y), (x+w, y+h), 255, -1)

    return filt_bboxes

def fill_small_contours(mask):
    valid_mask = np.zeros(mask.shape[:2], dtype="uint8")
    contours, hierarchies = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    areas = np.array([])
    filt_cnt = []
    for contour, hier in zip(contours, hierarchies[0]):  
        # Only consider contours of the lowest hierarchy
        if hier[-2] == -1:
            filt_cnt.append(contour)
            areas = np.append(areas, cv2.contourArea(contour))

    # Set a threshold based off mean contour size
    try:
        if np.average(areas) / np.std(areas) < 2:
            # Mean is comparable to std dev, threshold out things half a std dev away from mean
            idx = np.where(areas < np.average(areas)-np.std(areas)*0.5)[0]
            filt_cnt = np.array(filt_cnt)[idx]
            cv2.drawContours(valid_mask, filt_cnt, -1, 255, -1)
        # Otherwise do nothing
    except:
        print('No contours')
        return mask

    return cv2.bitwise_or(valid_mask, mask)

# Function that takes in a mask with ROI and linewidth and does a second pass to identify 
# smaller contours/components within. 
def decompose_contour(mask, og_rect, line_width):
    rects = np.array([0,0,0,0])
    x0,y0,w0,h0 = og_rect
    pad = 4*line_width
    mask2 = mask.copy()
    # 
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_width, line_width))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_width*2, line_width*2))
    # ymin, ymax, xmin, xmax
    mask_coords = [int(y0-0.15*h0), int(y0+1.15*h0), int(x0-0.15*w0), int(x0+1.15*w0)]
    roi = mask[int(y0-0.15*h0):int(y0+1.15*h0), int(x0-0.15*w0):int(x0+1.15*w0)]
    roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    roi_close = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel2)
    contours, _ = cv2.findContours(roi_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    roi_fill = roi_close.copy()
    if len(contours) > 1:
        # Likely multiple components in this ROI. Fill in these holes
        areas = np.array([])
        for contour in contours: 
            areas = np.append(areas, cv2.contourArea(contour))
        print('Multiple contours detected', areas, w0*h0/2, len(np.where(areas > 300)[0]))
        if len(np.where(areas > 300)[0]) == 2:
            for contour in contours:
                if cv2.contourArea(contour) < w0*h0/1.5:
                    print("FILLING")
                    cv2.drawContours(roi_fill, [contour], -1, 255, -1)
        elif len(np.where(areas > 300)[0]) > 2:
            for contour in contours:
                if cv2.contourArea(contour) < w0*h0/6:
                    print("FILLING")
                    cv2.drawContours(roi_fill, [contour], -1, 255, -1)
   
    roi_open = cv2.morphologyEx(roi_fill, cv2.MORPH_OPEN, kernel1)
    

    contours, _ = cv2.findContours(roi_open, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    print('AFTER >>> Num contours:', len(contours))
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        rects = np.vstack( (rects, np.array([mask_coords[2]+int(x)-pad, mask_coords[0]+int(y)-pad, w, h])) )
    #     cv2.rectangle(roi, (int(x), int(y)), (int(x+w), int(y+h)), 255, 2)
    #     cv2.rectangle(mask2, (mask_coords[2]+int(x)-pad,mask_coords[0]+int(y)-pad), \
    #         (mask_coords[2]+int(x+w)-pad,mask_coords[0]+int(y+h)-pad), 255, 2)
    #     cv2.rectangle(mask2, (mask_coords[2],mask_coords[0]), (mask_coords[3], mask_coords[1]), 255, 2)
        
    # cv2.imshow('Fill', roi_fill)
    # cv2.imshow('ROI', roi)
    # cv2.imshow('Closed up', roi_close)
    # cv2.imshow('Opened up', roi_open)
    # cv2.waitKey(0)

    return rects

def view_contours(mask):
    img = np.zeros(mask.shape[:2], dtype="uint8")
    contours, hiers = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    print('Hierarchy array:')
    print(hiers)
    for contour, hier in zip(contours, hiers[0]): 
        x,y,w,h = cv2.boundingRect(contour)
        temp = img.copy()
        cv2.drawContours(temp, [contour], -1, 255, 1)
        cv2.putText(temp, str(hier), (5,25), cv2.FONT_HERSHEY_COMPLEX, 1, 255)
        cv2.imshow('contour', temp)
        cv2.waitKey(0)
