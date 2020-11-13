# Find direction of each element

# Import required packages
import cv2 
import numpy as np

# This class finds the direction of each detected element, it takes an image as 
# an input and retuns the rotation of component
class DirectionIdentification(object):
    def __init__(self):
        # Initialise the height, width, image and rotation index
        self._img_height = []
        self._img_width = []
        self._img = []
        self._rot_idx = []
        

    # This function reads and preprocesses the image
    def preprocess_image(self,img):
        # Get image, its dimensions and a fourth of the dimensions
        self._img = img
        self._img_height,self._img_width = self._img.shape[:2]
        self._width_div_4 = int(self._img_width/4)
        self._height_div_4 = int(self._img_height/4)

        # Blur image to get rid of noise and get a binary image
        # MAYBE YOU CAN GET A BINARY IMAGE FROM DANIEL
        self._blur = cv2.GaussianBlur(self._img, (5,5), 0)
        self._bw = cv2.adaptiveThreshold(self._blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                 cv2.THRESH_BINARY_INV, 11, 2)

    # Find the direction of a resistor by comparing the height and width of 
    # the bounding box    
    def find_resistor_direction(self):
        if (self._img_height > self._img_width):
            print('----------- Vertical inductor/resistor -----------')
            self._rot_idx = 0
        else:
            print('----------- Horizontal inductor/resistor -----------')
            self._rot_idx = 1

        return self._rot_idx

    # Find the direction of the inductor the same way a resistor is found
    def find_inductor_direction(self):
        self._rot_idx = self.find_resistor_direction()
        return self._rot_idx

    # Find the horizontal lines at the end of the image which would correspond
    # to the endpoints of a component
    def get_horizontal_contours(self):

        # Copy the binary image
        horizontal = np.copy(self._bw)

        # Initialise flags 
        horz_left = 0
        horz_right = 0

        # Specify size on horizontal axis
        cols = horizontal.shape[1]
        horizontal_size = cols // 10

        # Create structure element for extracting horizontal lines through morphology operations
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        
        # Apply morphology operations
        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure)
        
        # Show extracted horizontal lines
        # cv2.imshow("horizontal", horizontal)
        # cv2.waitKey(0)

        # Find the contours
        contours, _ = cv2.findContours(horizontal, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        # Loop through each contour 
        for c in contours:

            # Calculate moments for each contour
            M = cv2.moments(c)

            # If the line is too thin, skip 
            if (M["m00"]<1):
                break

            # Calculate center x coordinate of the contour
            cX = int(M["m10"] / M["m00"])

            # Check if the center x is towards the left of the image
            if (cX < self._height_div_4):
                print('----------- Horizontal line on the left -----------')
                horz_left = 1

            # Check if the center x is towards the right of the image
            elif (cX > 3*self._height_div_4):
                print('----------- Horizontal line on the right -----------')
                horz_right =1

        # Return detection 
        return [horizontal, horz_left,horz_right]

    # Find the vertical lines at the end of the image which would correspond
    # to the endpoints of a component
    def get_vertical_contours(self):
        
        # Copy the binary image
        vertical = np.copy(self._bw)

        # Initialise flags 
        vert_top =0
        vert_bottom =0

        # Specify size on vertical axis
        rows = vertical.shape[0]
        verticalsize = rows // 10

        # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        
        # Apply morphology operations
        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)
        
        # Show extracted vertical lines
        # cv2.imshow("vertical", vertical)
        # cv2.waitKey(0)

        # Find the contours
        contours, _ = cv2.findContours(vertical, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        # Loop through each contour 
        for c in contours:

            # Calculate moments for each contour
            M = cv2.moments(c)

            # If the line is too thin, skip 
            if (M["m00"]<1):
                break

            # Calculate center y coordinate of the contour
            cY = int(M["m01"] / M["m00"])

            # Check if the center y is towards the top of the image
            if (cY < self._width_div_4 ):
                print('----------- Vertical line on the top -----------')
                vert_top =1

            # Check if the center y is towards the bottom of the image
            elif( cY > 3*self._width_div_4):
                print('----------- Vertical line on the bottom -----------')
                vert_bottom=1

        # Return detection 
        return [vertical,vert_top,vert_bottom]

    # Find the direction of the power supply/voltage source 
    def find_power_supply_direction(self):
        # Get endpoints of an element in the horizontal and vertical direction
        [horizontal,horz_left,horz_right] = self.get_horizontal_contours()
        [vertical,vert_top,vert_bottom] = self.get_vertical_contours()
        
        # Combine both the horizontal and vertical binary images to find the 
        # plus sign on the circuit component
        and_result = cv2.bitwise_and(horizontal,vertical)
        # cv2.imshow("together", and_result)
        # cv2.waitKey(0)

        # Get the middle of the image
        middle_results = and_result[self._height_div_4:3*self._height_div_4,self._width_div_4:3*self._width_div_4]
        # cv2.imshow("mid_Res", middle_results)
        # cv2.waitKey(0)

        # Calculate moments of middle image
        M_mid = cv2.moments(middle_results)
        
        # Calculate x,y coordinate of center
        cX_mid = int(M_mid["m10"] / M_mid["m00"]) + self._width_div_4
        cY_mid = int(M_mid["m01"] / M_mid["m00"]) + self._height_div_4

        # Check if the circuit component is vertical
        if((vert_top+vert_bottom)>(horz_left+horz_right)):

            # Check where the 'plus' sign is
            if (cY_mid < 2*self._width_div_4):
                print('Plus on top')
                self._rot_idx = 2
            else:
                print('Plus on bott')
                self._rot_idx = 0
            
        # The circuit element is horizontal
        else:
            # Check where the 'plus' sign is    
            if (cX_mid < 2*self._height_div_4):
                print('Plus on left')
                self._rot_idx = 1
            else:
                print('Plus on right')
                self._rot_idx = 3

        # Return the rotation index
        return self._rot_idx

    # Find the direction of the capacitor
    def find_capacitor_direction(self):

        # Get endpoints of an element in the horizontal and vertical direction
        [horizontal,horz_left,horz_right] = self.get_horizontal_contours()
        [vertical,vert_top,vert_bottom] = self.get_vertical_contours()

        # Check if the circuit component is vertical
        if((vert_top+vert_bottom)>(horz_left+horz_right)):
            print('Vertical')
            self._rot_idx = 0
        # The circuit element is horizontal
        else:
            print('Horizontal')
            self._rot_idx = 1

        # Return the rotation index
        return self._rot_idx

    def find_diode_direction(self):

        # Get endpoints of an element in the horizontal and vertical direction
        [_,horz_left,horz_right] = self.get_horizontal_contours()
        [_,vert_top,vert_bottom] = self.get_vertical_contours()

        # Calculate the moment of the whole binary image
        M_arrow = cv2.moments(self._bw)
        cX_arrow = int(M_arrow["m10"]  / M_arrow["m00"])
        cY_arrow = int(M_arrow["m01"]  / M_arrow["m00"])

        # Get the middle of the image
        middle_results = self._bw[self._height_div_4:3*self._height_div_4,self._width_div_4:3*self._width_div_4]

        # Calculate moments of middle image
        M_mid = cv2.moments(middle_results)
        
        # Calculate x,y coordinate of center
        cX_mid = int(M_mid["m10"] / M_mid["m00"]) + self._width_div_4
        cY_mid = int(M_mid["m01"] / M_mid["m00"]) + self._height_div_4

        # cv2.imshow('all',self._bw)
        # cv2.imshow("mid", middle_results)
        # cv2.waitKey(0)

        # Check if the circuit component is vertical
        if((vert_top+vert_bottom)>(horz_left+horz_right)):

            # Check if the moments shifted to the top which suggests the diode
            # is pointing up
            if(cY_arrow < cY_mid):
                print('Vertical diode to the top')
                self._rot_idx = 0
            else:
                print('Vertical diode to the bottom')
                self._rot_idx = 2
            # print('y shift')
            # print(cY_arrow,cY_mid)
            # print('x shift')
            # print(cX_arrow,cX_mid)

        # The circuit element is horizontal
        else:
            # Check if the moments shift to the left which suggests the diode
            # is pointing left
            if(cY_arrow < cY_mid):
                print('Horizontal diode to the left')
                self._rot_idx = 3
            else:
                print('Horizontal diode to the right')
                self._rot_idx = 1
            # print('y shift')
            # print(cY_arrow,cY_mid)
            # print('x shift')
            # print(cX_arrow,cX_mid)

        # Return the rotation index
        return self._rot_idx

    # Find the direction of the ground component
    def find_ground_direction(self):

        # Get endpoints of an element in the horizontal and vertical direction
        [_,horz_left,horz_right] = self.get_horizontal_contours()
        [_,vert_top,vert_bottom] = self.get_vertical_contours()

        # Check where the ground element connects to the circuit
        if(vert_top):
            print("Connected at top")
            self._rot_idx = 2
        elif(vert_bottom):
            print("Connected at bottom")
            self._rot_idx = 0
        elif(horz_left):
            print("Connected at left")
            self._rot_idx = 1
        elif(horz_right):
            print("Connected at right")
            self._rot_idx = 3

        # Return the rotation index
        return self._rot_idx

        




     

