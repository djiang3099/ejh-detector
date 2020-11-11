import cv2
import numpy as np
import os
import csv
path = 'scale'
with open('large.csv','w') as f1:
    writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
    for image_path in os.listdir(path):

        # create the full input path and read the file
        input_path = os.path.join(path, image_path)


        imgGray = cv2.imread(input_path,0)
        size = np.size(imgGray)
        skel = np.zeros(imgGray.shape,np.uint8)

        ret,imgBase = cv2.threshold(imgGray,127,255,0)
        rows,cols = np.shape(imgBase)

        #Flooding the corners white
        cv2.floodFill(imgBase,None,(0,0), 255)
        cv2.floodFill(imgBase,None,(0,rows-1), 255)
        cv2.floodFill(imgBase,None,(cols-1,0), 255)
        cv2.floodFill(imgBase,None,(cols-1,rows-1), 255)

        #Invert
        imgBase=cv2.bitwise_not(imgBase)

        #Thinning
        skel=cv2.ximgproc.thinning(imgBase)

        cv2.namedWindow('skel',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('skel', 600,600)
        cv2.namedWindow('base',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('base', 600,600)
        cv2.imshow("skel",skel)
        cv2.imshow("base",imgBase)
        #scale value
        scale=((np.concatenate(imgBase).sum()-np.concatenate(skel).sum())/np.concatenate(skel).sum())
        print("Area ratio?? ",np.concatenate(skel).sum(),np.concatenate(imgBase).sum(),scale)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        writer.writerow([image_path,scale])
