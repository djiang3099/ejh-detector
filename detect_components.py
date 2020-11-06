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

def get_mask(img):
    # Gaussian Blur
    blur = cv2.GaussianBlur(img, (5,5), 0)
    blur = cv2.GaussianBlur(blur, (5,5), 0)

    # Adaptive thresholding to binarise the image
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY_INV, 11, 2)
    # cv2.imshow('thr', th3)            
    # th3 = cv2.bitwise_not(th3)
    
    th3 = cv2.copyMakeBorder(th3, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[0,0,0])
    

    # Closing operation to fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel, iterations=2)      ##### TODO: Maybe iterations needs to be tuned
    # closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    # closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)

    # Erosion to get rid of noise
    closing = cv2.erode(closing, kernel, iterations=1)
    return closing

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
    # cv2.imshow('closing', closing)
    # cv2.waitKey(0)
    
    # Canny edge detection 
    canny = cv2.Canny(closing, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    canny_dil = cv2.dilate(canny, kernel)   # Dilate to join them up 

    # Identify ROI for circuit and crop it out, TODO: Use 4sided polygon
    contours, _ = cv2.findContours(canny_dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    best_contour = None
    for contour in contours: 
        if best_contour is None:
            best_contour = contour
        elif cv2.contourArea(contour) > cv2.contourArea(best_contour):
            best_contour = contour

    # cv2.imshow('canny_dil', canny_dil)
    # print('Press any key to continue...')
    # cv2.waitKey(0)

    # Get the min enclosing rectangle, should contain the full circuit
    rect = cv2.minAreaRect(best_contour)
    box = np.int0(cv2.boxPoints(rect))  # Box points are clockwise starting from the lowest one
    crop_margin = 0.1
    big_box = enlarge_rotated_rect(box, crop_margin)
    cv2.drawContours(markup, [big_box], 0, (0,0,255), 2)

    # Extract this ROI and straighten it
    roi_width = int((1+crop_margin)*rect[1][0])
    roi_height = int((1+crop_margin)*rect[1][1])
    cropped = crop_n_rotate(img_pad, big_box, (roi_width, roi_height))

    # Rescale the cropped image to normalise
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

    # Get the relative scale of the circuit components
    canny_crop = cv2.Canny(thr_crop, 100, 200)
    scale = cv2.countNonZero(canny_crop) / cv2.countNonZero(thr_crop)
    print(scale)

    big_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (35,35))

    # Closing operation to turn components into blobs
    mask = cv2.morphologyEx(thr_crop, cv2.MORPH_CLOSE, big_kern)
    cv2.imshow('mask', mask)
    # cv2.waitKey(0)

    # Med erosion to get rid of lines
    init_cont = len(cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2])
    max_cont = init_cont
    peaked = False
    conts = []
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
            break
        # cv2.imshow('mask', mask2)
        # cv2.waitKey(0)

    if not peaked:
        line_width = conts.index(max(conts)) 
        print('Peak asssumed to be at', line_width)
    else:
        line_width = i
    line_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_width,line_width))
    mask3 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, line_kern)
    cv2.imshow('mask3', mask3)
    # cv2.waitKey(0)
    
    # Med dilation to get the component shapes back up to size
    mask = cv2.dilate(mask3, line_kern, iterations=int(2**(12/line_width)+4))
    cv2.imshow('mask', mask)
    # cv2.waitKey(0)

    # Get the bounding boxes for all the remaining components
    bboxes = np.array([0,0,0,0])
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        x = x-20
        y = y-20

        # Reject bboxes based on aspect ratio
        if 3 > w/h > 0.33:
            # Blow up the contour a bit to make sure all components are fully captured
            cv2.rectangle(cropped, (int(x-0.1*w), int(y-0.1*h)), (int(x+1.1*w), int(y+1.1*h)), (0,255,0), 2)
            cv2.putText(cropped, str(cv2.contourArea(contour)), (x,y+h), cv2.FONT_HERSHEY_COMPLEX, 1, (0,70,255))

            # Store the bbox
            bboxes = np.vstack( (bboxes, np.array([x,y,w,h])) )

    # cv2.imshow('image', cv2.hconcat([gray, canny, closing, mask]))
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
