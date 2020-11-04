import cv2
import numpy as np

def enlarge_rotated_rect(box, scale=0.05):
    pt1 = box[0]
    pt2 = box[1]
    pt3 = box[2]
    pt4 = box[3]
    big_box = np.zeros((4,2))
    big_box[0] = [pt1[0] + (pt1[0]-pt3[0])*scale, pt1[1] + (pt1[1]-pt3[1])*scale]
    big_box[2] = [pt3[0] - (pt1[0]-pt3[0])*scale, pt3[1] - (pt1[1]-pt3[1])*scale]
    big_box[1] = [pt2[0] + (pt2[0]-pt4[0])*scale, pt2[1] + (pt2[1]-pt4[1])*scale]
    big_box[3] = [pt4[0] - (pt2[0]-pt4[0])*scale, pt4[1] - (pt2[1]-pt4[1])*scale]
    return np.int0(big_box)

def crop_n_rotate(img, box, size)

def detect_components(img):
    # Resize the ima`ge to fit on the screen
    height, width = img.shape[:2]
    lim_dim = 550

    # Find which length is longer and scale it appropriately 
    if height > width:
        scale = width/lim_dim
    else:
        scale = height/lim_dim

    img = cv2.resize(img, ( int(width/scale), int(height/scale) ), dst=img, interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Preprocess the image to extract components
    # TODO: Handle images where only part of the page is captured and there's not uniform background
    # Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    blur = cv2.GaussianBlur(blur, (5,5), 0)

    # Adaptive equalisation on the image to get more uniform lighting
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.bitwise_not(th3)
    th3 = cv2.copyMakeBorder(th3, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[0,0,0])

    # Closing operation to fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mid_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    big_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (35,35))
    closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)

    # Erosion to get rid of noise
    closing = cv2.erode(closing, kernel, iterations=1)

    # Canny edge detection 
    canny = cv2.Canny(closing, 100, 200)

    canny_dil = cv2.dilate(canny, kernel)

    # Estimate scale of image - TODO: Could also usse this step to find the region of interest
    scale = cv2.countNonZero(canny)/cv2.countNonZero(closing)
    _, contours, _ = cv2.findContours(canny_dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    best_contour = None

    for contour in contours: 
        if best_contour is None:
            best_contour = contour
            continue
        elif cv2.contourArea(contour) > cv2.contourArea(best_contour):
            best_contour = contour

    cv2.imshow('markup', canny_dil)
    # print('Press any key to continue...')
    # cv2.waitKey(0)

    # Get the min enclosing rectangle, should contain the full circuit
    rect = cv2.minAreaRect(best_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)  # Box points are clockwise starting from the lowest one
    print(box)
    big_box = enlarge_rotated_rect(box)
    print(big_box)
    cv2.drawContours(img, [big_box], 0, (0,0,255), 2)

    # Another big kernel closing operation 
    mask = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, big_kern)

    # Med erosion to get rid of lines
    mask = cv2.erode(mask, mid_kern, iterations=1)

    # Med dilation to get the component shapes back up to size
    mask = cv2.dilate(mask, mid_kern, iterations=1)

    # Store all the bboxes
    bboxes = np.array([0,0,0,0])

    # Get the bounding boxes for all the remaining components
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)

        # Reject bboxes based on aspect ratio
        if 3 > w/h > 0.33:
            # Blow up the contour a bit to make sure all components are fully captured
            cv2.rectangle(img, (int(x-0.1*w), int(y-0.1*h)), (int(x+1.1*w), int(y+1.1*h)), (0,255,0), 2)
            cv2.putText(img, str(cv2.contourArea(contour)), (x,y+h), cv2.FONT_HERSHEY_COMPLEX, 1, (0,70,255))

            # Store the bbox
            bboxes = np.vstack( (bboxes, np.array([x,y,w,h])) )

    # cv2.imshow('image', cv2.hconcat([gray, canny, closing, mask]))
    cv2.imshow('markup', img)
    print('Press any key to continue...')
    cv2.waitKey(0)
    return bboxes[1:]

if __name__ == '__main__':
    # Import images 
    for i in range(1,7):
        path = './Data/Circuits/' + str(i) + '.jpg'
        img = cv2.imread(path)
        bboxes = detect_components(img)
        # print(bboxes)