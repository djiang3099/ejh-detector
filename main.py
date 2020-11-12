# AMME4710 - EJH Detector 2020
# Circuit Digitaliser 

# 470205127
# 470355499
# 470425954

import numpy as np
import cv2

from detect_components import *
from circuitDecode import *

if __name__ == '__main__':
    # Import images 
    for i in range(3,7):
        path = './Data/Circuits/' + str(i) + '.jpg'
        print(i)
        img = cv2.imread(path)

        # Find the locations of components in the image as Bboxes
        bboxes, img, mask = detect_components(img)
        cv2.imshow('Main Mask', mask)
        # print('Press any key to continue...')
        cv2.waitKey(0)

        # Now have a nx4 array of bbox coords, and nx2 array of class labels and rotations
        #mask=img
        components = []   
        for x,y,w,h in bboxes:
            # Pass to CNN to classify/confirm components, classified bboxes
            #### SYLVIE GOES BRRRRRRR
            comp_img = img[x:x+w, y:y+w]
            components.append(ComponentClass('r',0,(x,y),(x+w,y+h),mask))
        '''# creating list        
        components = []  
        
        # appending instances to list 
        components.append(ComponentClass('v',0,(575,167),(746,352),mask))
        components.append(ComponentClass('r',0,(306,239),(452,412),mask))
        components.append(ComponentClass('r',0,(0,216),(153,415),mask))
        '''
        components_type_export,adj_matrix=circuit_decode(mask,components)
        circuit_plot(components_type_export,adj_matrix)