# AMME4710 - EJH Detector 2020
# Circuit Digitaliser 

# 470205127
# 470355499
# 470425954

# Import required packages
import cv2 
import numpy as np

# This class finds the direction of each detected element, it takes an image as 
# an input and retuns the rotation of component
class DirectionIdentification(object):
    
    # Initialise class
    def __init__(self):

        # Initialise the height, width, image and rotation index
        self._img_width = []
        self._img_height = []
        self._img = []
        self._rot_idx = []
        
    # Read the image and preprocess it
    def preprocess_image(self,img):

        # Get image
        self._img = img

        # Fix the width of the image to 100 and scale the horizontal height to 
        # maintain aspect ratio
        width = 100
        scale_percent = int(100*width/img.shape[1]) # percent of original size
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # Resize image
        self._img_resize = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        # Blur image to get rid of noise and get a binary image
        self._blur = cv2.GaussianBlur(self._img_resize, (3,3), 0)
        self._bw = cv2.adaptiveThreshold(self._blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                 cv2.THRESH_BINARY_INV, 11, 2)

        # Define the image dimensions
        self._img_height,self._img_width = self._img_resize.shape[:2]

        # Define a third and quarter for later region extraction
        self._height_div_4 = int(self._img_height/4)
        self._width_div_4 = int(self._img_width/4)
        self._height_div_3 = int(self._img_height/3)
        self._width_div_3 = int(self._img_width/3)

    # Find the direction of a resistor by comparing the height and width of 
    # the bounding box    
    def find_resistor_direction(self):
        if (self._img_width < self._img_height):
            print('----------- Vertical inductor/resistor -----------')
            self._rot_idx = 0
        else:
            print('----------- Horizontal inductor/resistor -----------')
            self._rot_idx = 1
        return self._rot_idx

    # Find the direction of the inductor the same way a resistor is found
    def find_inductor_direction(self):
        return self.find_resistor_direction()

    # Find the horizontal lines at the left and right of an image which would 
    # correspond to the endpoints of a component
    def get_horizontal_contours(self):

        # Copy the binary image
        horizontal = np.copy(self._bw)

        # Initialise flags 
        horz_left = 0
        horz_right = 0

        # Specify size on horizontal axis
        cols = horizontal.shape[1]
        horizontal_size = cols // 7 

        # Create structure element for extracting horizontal lines through morphology operations
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        
        # Apply morphology operations
        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure)
        
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
            if (cX < self._width_div_4):
                print('----------- Horizontal line on the left -----------')
                horz_left = 1

            # Check if the center x is towards the right of the image
            elif (cX > 3*self._width_div_4):
                print('----------- Horizontal line on the right -----------')
                horz_right =1

        # Return detection 
        return [horizontal, horz_left,horz_right]

    # Find the vertical lines at top and bottom of the image which would 
    # correspond to the endpoints of a component
    def get_vertical_contours(self):
        
        # Copy the binary image
        vertical = np.copy(self._bw)

        # Initialise flags 
        vert_top =0
        vert_bottom =0

        # Specify size on vertical axis
        rows = vertical.shape[0]
        verticalsize = rows // 7

        # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        
        # Apply morphology operations
        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)
        
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
            if (cY < self._height_div_4 ):
                print('----------- Vertical line on the top -----------')
                vert_top =1

            # Check if the center y is towards the bottom of the image
            elif( cY > 3*self._height_div_4):
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

        # Get the middle of the image
        middle_results = and_result[self._width_div_4:3*self._width_div_4,self._height_div_4:3*self._height_div_4]

        # Find contours
        contours, _ = cv2.findContours(middle_results, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        c = max(contours, key = cv2.contourArea)

        # Calculate moments of middle image
        M_mid = cv2.moments(c)
        
        # Calculate x,y coordinate of center
        cX_mid = int(M_mid["m10"] / M_mid["m00"]) + self._height_div_4
        cY_mid = int(M_mid["m01"] / M_mid["m00"]) + self._width_div_4

        # Check if the circuit component is vertical
        if((vert_top+vert_bottom)>(horz_left+horz_right)):

            # Check where the 'plus' sign is
            if (cY_mid < 2*self._height_div_4):
                print('Plus on top')
                self._rot_idx = 2
            else:
                print('Plus on bott')
                self._rot_idx = 0
            
        # The circuit element is horizontal
        else:
            # Check where the 'plus' sign is    
            if (cX_mid < 2*self._width_div_4):
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

        # Check if the circuit component is vertical
        if((vert_top+vert_bottom)>(horz_left+horz_right)):
            
            # Get the middle of the image
            middle_results = self._bw[self._width_div_4:3*self._width_div_4,:]

            # Find the dimensions of the middle region
            [h_mid,w_mid] = middle_results.shape[0:2]

            # Separate the middle region into quadrants
            top_left = middle_results[1:int(h_mid/2),1:int(w_mid/2)]
            top_right = middle_results[1:int(h_mid/2),int(w_mid/2):w_mid]
            bot_left = middle_results[int(h_mid/2):h_mid,1:int(w_mid/2)]
            bot_right = middle_results[int(h_mid/2):h_mid,int(w_mid/2):w_mid]

            # List of all sections
            each_section = [top_left,top_right,bot_left,bot_right]
            each_section_x_coord = np.array([0,0,0,0])

            # Loop through each quadrant
            for i in range(0,len(each_section)):

                # Find contours
                contours, _ = cv2.findContours(each_section[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

                # Find the maximum contour
                c = max(contours, key = cv2.contourArea)

                # Calculate moments 
                M_mid = cv2.moments(c)
                           
                # Calculate x,y coordinate of center
                cX_mid = int(M_mid["m10"] / M_mid["m00"]) 
                cY_mid = int(M_mid["m01"] / M_mid["m00"]) 

                # Place the x coordinate in an array
                each_section_x_coord[i] = cX_mid
                # print(cX_mid,cY_mid)

            # Find the difference between the y coordinates of the top two 
            # quadrants. Do the same for the bottom quadrants
            top_x_values = (each_section_x_coord[1]-each_section_x_coord[0])
            bot_x_values = (each_section_x_coord[3]-each_section_x_coord[2])
            # print(top_y_values,bot_y_values)

            # Check if the top difference is less than the bottom difference, 
            # if it yes, then the diode points up
            if(top_x_values < bot_x_values):
                print('Vertical diode to the top')
                self._rot_idx = 0
            else:
                print('Vertical diode to the bottom')
                self._rot_idx = 2

        # The circuit element is horizontal
        else:

            # Get the middle of the image
            middle_results = self._bw[:,self._height_div_4:3*self._height_div_4]
            [h_mid,w_mid] = middle_results.shape[0:2]

            # Separate the middle region into quadrants
            top_left = middle_results[1:int(h_mid/2),1:int(w_mid/2)]
            top_right = middle_results[1:int(h_mid/2),int(w_mid/2):w_mid]
            bot_left = middle_results[int(h_mid/2):h_mid,1:int(w_mid/2)]
            bot_right = middle_results[int(h_mid/2):h_mid,int(w_mid/2):w_mid]

            # List of all sections
            each_section = [top_left,top_right,bot_left,bot_right]
            each_section_y_coord = np.array([0,0,0,0])

            # Loop through each quadrant
            for i in range(0,len(each_section)):

                # Find contours
                contours, _ = cv2.findContours(each_section[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

                # Find the biggest contour
                c = max(contours, key = cv2.contourArea)

                # Calculate moments 
                M_mid = cv2.moments(c)
          
                # Calculate x,y coordinate of center
                cX_mid = int(M_mid["m10"] / M_mid["m00"]) 
                cY_mid = int(M_mid["m01"] / M_mid["m00"]) 

                # Reccord the y coordinate
                each_section_y_coord[i] = cY_mid
                # print(cX_mid,cY_mid)

            # Find the difference between the x coordinates of the left  
            # quadrants. Do the same for the right quadrants
            left_y_values = (each_section_y_coord[0]-each_section_y_coord[2])
            right_y_values = (each_section_y_coord[1]-each_section_y_coord[3])
            # print(left_y_values,right_y_values)

            # Check if the right difference is less than the left difference, 
            # if it yes, then the diode points left
            if(left_y_values > right_y_values):
                print('Horizontal diode to the left')
                self._rot_idx = 3
            else:
                print('Horizontal diode to the right')
                self._rot_idx = 1

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

        




     

