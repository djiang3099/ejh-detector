import cv2
import numpy as np

# Import images 
img = cv2.imread('./Data/Circuits/7.jpg')

# Resize the image to fit on the screen
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

# Adaptive equalisation on the image to get more uniform lighting
th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY, 11, 2)
th3 = cv2.bitwise_not(th3)

# Closing operation to fill holes
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
mid_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
big_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (70,70))
closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)

# Erosion to get rid of noise
mask = cv2.erode(closing, kernel, iterations=1)

# Another big kernel closing operation 
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, big_kern)

# Med erosion to get rid of lines
mask = cv2.erode(mask, mid_kern, iterations=1)

# Med dilation to get the component shapes back up to size
mask = cv2.dilate(mask, mid_kern, iterations=1)

# Get the bounding boxes for all the remaining components
_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    # Blow up the contour a bit to make sure all components are fully captured
    cv2.rectangle(img, (int(x-0.1*w), int(y-0.1*h)), (int(x+1.1*w), int(y+1.1*h)), (0,255,0), 2)


cv2.imshow('image', cv2.hconcat([gray, th3, closing, mask]))
cv2.imshow('markup', img)
print('Press any key to continue...')
cv2.waitKey(0)