# AMME4710 - EJH Detector 2020
# Circuit Digitaliser 

# 470205127
# 470355499
# 470425954

import numpy as np
import cv2
import matplotlib.pyplot as plt

from detect_components import *
from circuitDecode import *
from component_classifier import *

if __name__ == '__main__':
    print('Running main')
    # Import images 
    for i in range(7,11):
        path = './Data/Circuits/' + str(i) + '.jpg'
        print(i)
        img = cv2.imread(path)

        # Find the locations of components in the image as Bboxes
        bboxes, img_2, mask = detect_components(img)
        gray = cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY)
        cv2.imshow('Main Mask', mask)
        cv2.imshow('Main Img', gray)
        # print('Press any key to continue...')
        cv2.waitKey(0)

        # Now have a nx4 array of bbox coords, and nx2 array of class labels and rotations
        #mask=img
        components = []   
        for x,y,w,h in bboxes:
            # Pass to CNN to classify/confirm components, classified bboxes
            #### SYLVIE GOES BRRRRRRR
            # int(x-0.1*w), int(y-0.1*h)), (int(x+1.1*w), int(y+1.1*h)
            comp_img = []
            comp_img = gray[int(y-0.15*h):int(y+1.15*h), int(x-0.15*w):int(x+1.15*w)]
            # cv2.imshow('gray', comp_img)
            # cv2.waitKey(0)
            imgplot = plt.imshow(comp_img)
            plt.show()

            [element,rot_idx] = component_classifier(comp_img)
            # if not(element=='x'):
            components.append(ComponentClass(element,rot_idx,(x,y),(x+w,y+h),mask))

            
            # components.append(ComponentClass('r', 0,(x,y),(x+w,y+h),mask))
        '''# creating list        
        components = []  
        
        # appending instances to list 
        components.append(ComponentClass('v',0,(575,167),(746,352),mask))
        components.append(ComponentClass('r',0,(306,239),(452,412),mask))
        components.append(ComponentClass('r',0,(0,216),(153,415),mask))
        '''
        components_type_export,adj_matrix=circuit_decode(mask,components)
        circuit_plot(components_type_export,adj_matrix)