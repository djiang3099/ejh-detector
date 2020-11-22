# AMME4710 - EJH Detector 2020
# Circuit Digitaliser 

# 470205127
# 470355499
# 470425954

import cv2
import numpy as np
from vis import circuit_plot


class ComponentClass:
    def __init__(self,type,rot,point1,point2,source_mask):
        #Define all the class properties
        self.type=type
        self.rot=rot
        self.__point1=np.asarray(point1)
        self.__point2=np.asarray(point2)
        #Scaled down bounding box
        __small_offset=np.asarray([3,3])
        self.small_point1=self.__point1+__small_offset
        self.small_point2=self.__point2-__small_offset

        #Make the mask for both pins (this can be extended to more pins in a child class)
        self.pin_rectangles=[]
        self.pin_rectangles.append(np.zeros_like(source_mask))
        self.pin_rectangles.append(np.zeros_like(source_mask))

        #Define the pin sizes based on the rotation of the component
        self.__generate_pin_boxes()
        #print("Direction is "+str(rot))
        if rot==None:
              raise Exception("No rotation given to the class") 
        #Debugging code to show both of the pins
        # cv2.imshow("positive",self.pin_rectangles[0])
        # cv2.waitKey(0)
        # cv2.imshow("negative",self.pin_rectangles[1])
        # cv2.waitKey(0)

    #Generate the pin boxes
    def __generate_pin_boxes(self): 
        #Positive Bottom
        '''
        -
        +
        '''
        if(self.rot==0):
            #Find the middle along the y axis
            midPointy=(self.__point2[1]+self.__point1[1])/2
            #Generate the top pin and bottom pin
            top1=self.__point1
            top2=np.asarray([self.__point2[0],midPointy])

            bottom1=np.asarray([self.__point1[0],midPointy])
            bottom2=self.__point2

            #Store so index 0 is always negative
            self.pin_rectangles[0]=cv2.rectangle(self.pin_rectangles[0],tuple(top1.astype(int)),tuple(top2.astype(int)), 255, -1) 

            self.pin_rectangles[1]=cv2.rectangle(self.pin_rectangles[1],tuple(bottom1.astype(int)),tuple(bottom2.astype(int)), 255, -1) 
            '''
            +-
            '''
        elif(self.rot==1):
            #Find the middle along the x axis
            midPointx=(self.__point2[0]+self.__point1[0])/2
            #Generate the left and right pin
            left1=self.__point1
            left2=np.asarray([midPointx,self.__point2[1]])

            right1=np.asarray([midPointx,self.__point1[1]])
            right2=self.__point2

            #Store so index 0 is always negative
            self.pin_rectangles[0]=cv2.rectangle(self.pin_rectangles[0],tuple(right1.astype(int)),tuple(right2.astype(int)), 255, -1) 

            self.pin_rectangles[1]=cv2.rectangle(self.pin_rectangles[1],tuple(left1.astype(int)),tuple(left2.astype(int)), 255, -1)            
            '''
            +
            -
            '''
        elif(self.rot==2):
            #Find the middle along the y axis
            midPointy=(self.__point2[1]+self.__point1[1])/2
            #Generate the top pin and bottom pin
            top1=self.__point1
            top2=np.asarray([self.__point2[0],midPointy])

            bottom1=np.asarray([self.__point1[0],midPointy])
            bottom2=self.__point2

            #Store so index 0 is always negative
            self.pin_rectangles[0]=cv2.rectangle(self.pin_rectangles[0],tuple(bottom1.astype(int)),tuple(bottom2.astype(int)), 255, -1) 

            self.pin_rectangles[1]=cv2.rectangle(self.pin_rectangles[1],tuple(top1.astype(int)),tuple(top2.astype(int)), 255, -1) 

            '''
            -+
            '''
        elif(self.rot==3):
            #Find the middle along the x axis
            midPointx=(self.__point2[0]+self.__point1[0])/2
            #Generate the left and right pin
            left1=self.__point1
            left2=np.asarray([midPointx,self.__point2[1]])

            right1=np.asarray([midPointx,self.__point1[1]])
            right2=self.__point2

            #Store so index 0 is always negative
            self.pin_rectangles[0]=cv2.rectangle(self.pin_rectangles[0],tuple(left1.astype(int)),tuple(left2.astype(int)), 255, -1)   

            self.pin_rectangles[1]=cv2.rectangle(self.pin_rectangles[1],tuple(right1.astype(int)),tuple(right2.astype(int)), 255, -1) 

    def check_intersect(self,line_mask):
        #Check both pins to see if they intersect the mask of the line
        for pin_index,pin_mask in enumerate(self.pin_rectangles):
            intersect=cv2.bitwise_and(line_mask,pin_mask)
            #Debug image display
            #cv2.imshow("or",cv2.bitwise_or(line_mask,pin_mask))
            #cv2.waitKey(0)

            #If we have more than one pixel then there is an intersection
            if np.concatenate(intersect).sum() >0:
                #If positive return 1
                return pin_index+1


        return None


def circuit_decode(mask,components):
    #Create a mask the same size of the image
    component_mask=np.zeros_like(mask)
    #Export the component type for later displaying
    components_type_export=[]

    #Loop through all the components and make one big mask with all the component bounding boxes
    for comp_idx,component in enumerate(components): 
        component_mask=cv2.rectangle(component_mask, tuple(component.small_point1), tuple(component.small_point2), 255, -1)
       # cv2.imshow("components",component_mask)
        components_type_export.append(component.type+str(comp_idx))
        #cv2.waitKey(0)
    
    #Remove the components from the circuit leaving behind the lines
    line_mask=cv2.bitwise_and(mask,cv2.bitwise_not(component_mask))

    # cv2.imshow("lines",line_mask)
    # cv2.waitKey(0)
    number_of_components=len(components)
    adj_matrix=np.empty((0,number_of_components), int)
    contours, _ = cv2.findContours(line_mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

    #Loop through all the line blobs and look for intersections
    for line_idx, line in enumerate(contours):
        #Redraw the line blob by itself
        line_mask=cv2.drawContours(np.zeros_like(mask),[line], -1, 255, cv2.FILLED)
        #cv2.imshow("contours",line_mask)
        #cv2.waitKey(0)
        connection_found=False
        #Loop through all the compoents
        for component_idx,component in enumerate(components):
            #print(component_idx)

            #Do we have an intersection
            pintype=component.check_intersect(line_mask)
            if pintype is not None:
                #We have a connection
                #Have we had a previous connection i.e. is the line in adj matrix
                if connection_found==False:
                    #Add the new line to the matrix
                    adj_matrix=np.append(adj_matrix, np.zeros((1,len(components))), axis=0)
                    connection_found=True

                #Store the connection type
                adj_matrix[-1,component_idx]=pintype
        #After checking all the components if we only have 1 one interection than it isn't a line
        #A line needs minimum 2 endpoints
        if not(adj_matrix.size==0):
            found_connect_idx=np.where(adj_matrix[-1]>0)[0]

            #All lines should have 2 connections
            if len(found_connect_idx) < 2:
                #Remove the line
                adj_matrix=np.delete(adj_matrix,-1,0)
            




    #Return the component types and the connections
    return components_type_export,adj_matrix
    
