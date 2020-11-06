from cv2 import cv2
import numpy as np
'''from dataclasses import dataclass

@dataclass'''
class ComponentClass:



    def __init__(self,type,rot,point1,point2,source_mask):
        self.type=type
        self.rot=rot
        self.__point1=np.asarray(point1)
        self.__point2=np.asarray(point2)
        __small_offset=np.asarray([3,3])
        self.small_point1=self.__point1+__small_offset
        self.small_point2=self.__point2-__small_offset
        self.pin_rectangles=[]
        self.pin_rectangles.append(np.zeros_like(source_mask))
        self.pin_rectangles.append(np.zeros_like(source_mask))
        cv2.imshow("circuit",source_mask)
        cv2.waitKey(0)
        self.__generate_pin_boxes()
        cv2.imshow("positive",self.pin_rectangles[0])
        cv2.waitKey(0)
        cv2.imshow("negative",self.pin_rectangles[1])
        cv2.waitKey(0)

    def __generate_pin_boxes(self): 
        #Positive Bottom
        if(self.rot==0):
            midPointy=(self.__point2[1]+self.__point1[1])/2
            top1=self.__point1
            top2=np.asarray([self.__point2[0],midPointy])

            bottom1=np.asarray([self.__point1[0],midPointy])
            bottom2=self.__point2

            #Index 0 is always negative
            self.pin_rectangles[0]=cv2.rectangle(self.pin_rectangles[0],tuple(top1.astype(int)),tuple(top2.astype(int)), 255, -1) 

            self.pin_rectangles[1]=cv2.rectangle(self.pin_rectangles[1],tuple(bottom1.astype(int)),tuple(bottom2.astype(int)), 255, -1) 
        elif(self.rot==1):
            midPointx=(self.__point2[0]+self.__point1[0])/2
            left1=self.__point1
            left2=np.asarray([midPointx,self.__point2[1]])

            right1=np.asarray([midPointx,self.__point1[1]])
            right2=self.__point2

             #Index 0 is always negative
            self.pin_rectangles[0]=cv2.rectangle(self.pin_rectangles[0],tuple(right1.astype(int)),tuple(right2.astype(int)), 255, -1) 

            self.pin_rectangles[1]=cv2.rectangle(self.pin_rectangles[1],tuple(left1.astype(int)),tuple(left2.astype(int)), 255, -1)            

        elif(self.rot==2):
            midPointy=(self.__point2[1]+self.__point1[1])/2
            top1=self.__point1
            top2=np.asarray([self.__point2[0],midPointy])

            bottom1=np.asarray([self.__point1[0],midPointy])
            bottom2=self.__point2

            #Index 0 is always negative
            self.pin_rectangles[0]=cv2.rectangle(self.pin_rectangles[0],tuple(bottom1.astype(int)),tuple(bottom2.astype(int)), 255, -1) 

            self.pin_rectangles[1]=cv2.rectangle(self.pin_rectangles[1],tuple(top1.astype(int)),tuple(top2.astype(int)), 255, -1) 

        elif(self.rot==4):
            midPointx=(self.__point2[0]+self.__point1[0])/2
            left1=self.__point1
            left2=np.asarray([midPointx,self.__point2[1]])

            right1=np.asarray([midPointx,self.__point1[1]])
            right2=self.__point2

             #Index 0 is always negative
            self.pin_rectangles[0]=cv2.rectangle(self.pin_rectangles[0],tuple(left1.astype(int)),tuple(left2.astype(int)), 255, -1)   

            self.pin_rectangles[1]=cv2.rectangle(self.pin_rectangles[1],tuple(right1.astype(int)),tuple(right2.astype(int)), 255, -1) 

    def check_intersect(self,line_mask):
        for pin_index,pin_mask in enumerate(self.pin_rectangles):
            #print(pin_index)
            intersect=cv2.bitwise_and(line_mask,pin_mask)
            cv2.imshow("or",cv2.bitwise_or(line_mask,pin_mask))
            cv2.waitKey(0)
            if np.concatenate(intersect).sum() >0:
                #If positive return 1
                return pin_index+1
                '''if pin_index==0:
                    return 1
                else:
                    return -1'''

        return None
            #np.concatenate(skel).sum()
        #print("Private method") 
'''
def adj_builder(mask,components):
    #Create a mask with just the lines

    #Scale up compoents by small amount 

    #Loop through lines

        #Store temp intersetions
        #store connection index and +,- if negative or positve


        #if no intersection don't add this line to adj
        #or if only one intersection invalid
'''





if __name__ == '__main__':
    # Import images 
    imgGray=cv2.imread('custom_mask.jpg', cv2.IMREAD_GRAYSCALE)
    ret,mask = cv2.threshold(imgGray,127,255,cv2.THRESH_BINARY)
    cv2.imshow("mask",mask)
    cv2.waitKey(0)

    # creating list        
    components = []  
    
    # appending instances to list 
    components.append(ComponentClass('v',1,(575,167),(746,352),mask))
    components.append(ComponentClass('r',1,(306,239),(452,412),mask))
    components.append(ComponentClass('r',1,(0,216),(153,415),mask))


    #Make a mask for components
    component_mask=np.zeros_like(mask)

    for component in components: 
        #print( obj.name, obj.roll, sep =' ' ) 
        component_mask=cv2.rectangle(component_mask, tuple(component.small_point1), tuple(component.small_point2), 255, -1)
        cv2.imshow("components",component_mask)
        cv2.waitKey(0)
    
    line_mask=cv2.bitwise_and(mask,cv2.bitwise_not(component_mask))

    cv2.imshow("lines",line_mask)
    cv2.waitKey(0)
    number_of_components=len(components)
    adj_matrix=np.zeros((number_of_components,number_of_components))

    contours, hierarchy = cv2.findContours(line_mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    for line_idx, line in enumerate(contours):
        #print(line_idx)
        #If you dont' do [cnt] it takes cnt to be a list of sub contours ie.e. each pixel on the edge is a contour
        #filling a pixel doesn't make sense
        line_mask=cv2.drawContours(np.zeros_like(mask),[line], -1, 255, cv2.FILLED)
        cv2.imshow("contours",line_mask)
        cv2.waitKey(0)
        connection_found=False
        for component_idx,component in enumerate(components):
            print(component_idx)
            pintype=component.check_intersect(line_mask)
            if pintype is not None:
                #We have a connection
                #Have we had a previous connection i.e. is the line in adj matrix
                if connection_found==False:
                    adj_matrix=np.pad(adj_matrix,(0,1))
                    connection_found=True
                #Have we already added a row
                
                adj_matrix[component_idx,number_of_components+line_idx-1]=pintype
                adj_matrix[number_of_components+line_idx-1,component_idx]=pintype
                #might just have an array, each row is the cocmponent and each col is the line
                #or vise versa
                

    print(adj_matrix)
        # If valid line
        
    '''
    components.append(ComponentClass('Akash',2)) 
    components.append(geeks('Deependra',40)) 
    components.append(geeks('Reaper',44))
    '''



    #component_mask=cv.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]	)
    cv2.destroyAllWindows()