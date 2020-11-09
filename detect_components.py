import cv2
import numpy as np
import time

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

def remove_noise(img):
    mask = np.zeros(img.shape[:2], dtype="uint8")
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    areas = np.array([])
    for contour in contours: 
        areas = np.append(areas, cv2.contourArea(contour))

    avg = np.mean(areas)
    idx = np.where(areas > 300)[0]
    filt_cnt = np.array(contours)[idx]
    cv2.drawContours(mask, filt_cnt, -1, 255, 1)

    return mask

def get_mask(img):
    # Gaussian Blur
    blur = cv2.GaussianBlur(img, (5,5), 0)
    blur = cv2.GaussianBlur(blur, (5,5), 0)

    # Adaptive thresholding to binarise the image
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY_INV, 11, 2)
    
    th3 = cv2.copyMakeBorder(th3, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[0,0,0])
    
    # Closing operation to fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel, iterations=2)

    closing = cv2.erode(closing, kernel, iterations=1)
    cv2.imshow('thr', closing)
    # Erosion to get rid of noise
    closing = remove_noise(closing)

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
    if np.average(filt_areas) / np.std(filt_areas) > 1.3:
        # There are few outliers, reject two std dev away from avg
        idx = np.where(areas > np.average(filt_areas)-np.std(filt_areas))[0]
    else: 
        # Mean is similar to std dev, meaning significant number of outliers
        idx = np.where(areas > np.average(filt_areas)-np.std(filt_areas)*0.5)[0]
    # Keep the contours specified by idx
    filt_cnt = np.array(contours)[idx]

    for contour in filt_cnt:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(valid_mask, (x,y), (x+w, y+h), 255, -1)

    return cv2.bitwise_and(valid_mask, mask)

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
    if np.average(areas) / np.std(areas) < 2:
        # Mean is comparable to std dev, threshold out things half a std dev away from mean
        idx = np.where(areas < np.average(areas)-np.std(areas)*0.5)[0]
        filt_cnt = np.array(filt_cnt)[idx]
        cv2.drawContours(valid_mask, filt_cnt, -1, 255, -1)
    # Otherwise do nothing

    return cv2.bitwise_or(valid_mask, mask)

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

def detect_components(img):
    # Resize the image to fit on the screen
    height, width = img.shape[:2]
    lim_dim = 550

    # Find which length is longer and scale it appropriately 
    if height > width:
        scale = width/lim_dim
    else:
        scale = height/lim_dim

    img = cv2.resize(img, ( int(width/scale), int(height/scale) ), dst=img, interpolation = cv2.INTER_CUBIC)
    img_pad = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_DEFAULT)
    markup = img_pad.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Preprocess the image to extract components
    # TODO: Handle images where only part of the page is captured and there's not uniform background
    closing = get_mask(gray)
    
    # Canny edge detection 
    canny = cv2.Canny(closing, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    canny_dil = cv2.dilate(canny, kernel)   # Dilate to join them up 

    # Crop the circuit out of the image and rotate to straighten
    cropped = extract_circuit(img_pad, canny_dil, crop_margin=0.1)

    # Rescale the cropped image to normalise
    roi_height, roi_width = cropped.shape[:2]
    if roi_height > roi_width:
        scale = roi_width/lim_dim
    else:
        scale = roi_height/lim_dim

    cropped = cv2.resize(cropped, ( int(roi_width/scale), int(roi_height/scale) ), interpolation = cv2.INTER_CUBIC)
    # cv2.imshow('cropped', cropped)

    ##### Perform component extraction on this cropped image #####
    # Gaussian Blur
    gray_crop = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    thr_crop = get_mask(gray_crop)
    cv2.imshow('thr cropped', thr_crop)

    big_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (35,35))

    # Closing operation to turn components into blobs
    mask = cv2.morphologyEx(thr_crop, cv2.MORPH_CLOSE, big_kern)
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
                print("Peaked!")
            max_cont = len(contours)
        elif (max_cont+init_cont)/2 >= len(contours) and peaked:
            print("Peak found at kernel size", i)
            found = True
            break
        cv2.imshow('mask', mask2)
        # cv2.waitKey(0) 

    # Not found when the circuit is dense
    if not found:
        line_width = (conts.index(max(conts)) ) *2
        print('Peak asssumed to be at', line_width)
    else:
        line_width = i*2

    # Do it all over again, moderated by the linewidth
    big_line_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20+2*line_width,20+2*line_width))
    line_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_width,line_width))

    # Closing operation to turn components into blobs
    mask_line = cv2.morphologyEx(thr_crop, cv2.MORPH_CLOSE, big_line_kern)
    # cv2.imshow('mask_line', mask_line)
    # cv2.waitKey(0)

    mask_line = fill_small_contours(mask_line)
    cv2.imshow('mask_line', mask_line)
    cv2.waitKey(0)

    mask_line = cv2.morphologyEx(mask_line, cv2.MORPH_OPEN, line_kern)
    cv2.imshow('mask_line', mask_line)
    cv2.waitKey(0)
    
    # Remove outlier blobs generated from line intersections
    mask_line = reject_outliers(mask_line)

    # Med dilation to get the component shapes back up to size
    mask_line = cv2.dilate(mask_line, line_kern, iterations=1)
    cv2.imshow('mask_line', mask_line)
    cv2.waitKey(0)

    # Get the bounding boxes for all the remaining components
    bboxes = np.array([0,0,0,0])
    contours, _ = cv2.findContours(mask_line, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        # Shift it back to account for the padding
        x = x-20
        y = y-20

        # Reject bboxes based on aspect ratio
        if 3 > w/h > 0.33:
            # Blow up the contour a bit to make sure all components are fully captured
            cv2.rectangle(cropped, (int(x-0.1*w), int(y-0.1*h)), (int(x+1.1*w), int(y+1.1*h)), (0,255,0), 2)
            cv2.putText(cropped, str(cv2.contourArea(contour)), (x,y+h), cv2.FONT_HERSHEY_COMPLEX, 1, (0,70,255))

            # Store the bbox
            bboxes = np.vstack( (bboxes, np.array([x,y,w,h])) )

    cv2.imshow('cropped', cropped)
    print('Press any key to continue...')
    cv2.waitKey(0)
    return bboxes[1:]

if __name__ == '__main__':
    # Import images 
    for i in range(1,9):
        path = './Data/Circuits/' + str(i) + '.jpg'
        img = cv2.imread(path)
        bboxes = detect_components(img)
    # names = ['5','10','15','15manny','20','25','35','40','40many','60','70','70_2','80','80_2','80_3','90','90_2']
    # for i in range(1, len(names)):
    #     path = './scale/' + names[i] + '.jpg'
    #     img = cv2.imread(path)
    #     bboxes = detect_components(img)
