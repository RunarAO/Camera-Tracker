#!/usr/bin/env python
import cv2
import argparse
import glob
#from skimage.io import imread
import numpy as np
import rospy
import os
import time
import sys
import math
import tf

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))

import autosea_msgs.msg as automsg
import std_msgs.msg as stdmsg
import geometry_msgs.msg as geomsg
identity_quat = geomsg.Quaternion(0,0,0,1)
identity_pos = geomsg.Point(0,0,0)
identity_pose = geomsg.Pose(position=identity_pos, orientation=identity_quat)

from re3.tracker import re3_tracker
from re3.tracker.darknet import darknet_orig as dn

from re3.re3_utils.util import drawing
from re3.re3_utils.util import bb_util
from re3.re3_utils.util import im_util

from re3.constants import OUTPUT_WIDTH
from re3.constants import OUTPUT_HEIGHT
from re3.constants import PADDING

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)


boxToDraw = np.zeros(4)
initialize = True


def show_webcam(image, corners=None):
    global initialize, boxToDraw#,tracker

    if corners is None:
        try:
            boxToDraw = tracker.track(image[:,:,::-1], 'Cam')
        except:
            print("No Bbox to track")

    else:# all(corners)!=0:
        iou = bb_intersection_over_union(boxToDraw,corners)
        if iou < 0.3:
            initialize = True
            print ("UPDATED")

    if initialize and corners is not None:
    	if all(corners)!=0:
    		boxToDraw = corners
    		initialize = False
    		boxToDraw = tracker.track(image[:,:,::-1], 'Cam', boxToDraw)

    elif ((abs(boxToDraw[0]-boxToDraw[2]) > 5) and (abs(boxToDraw[1]-boxToDraw[3]) > 5)):
    	boxToDraw = tracker.track(image[:,:,::-1], 'Cam')
    	cv2.rectangle(im,
	        (int(boxToDraw[0]), int(boxToDraw[1])),
	        (int(boxToDraw[2]), int(boxToDraw[3])),
	        [0,0,255], 2)

    #else:
    #    print("TERMINATED")
    
    cv2.imshow('Cam', im)
    #print (boxToDraw)


def yolo_boxes_to_corners(bbox):
    """Convert xc, yc, w, h to xmin, ymin, xmax, ymax along last axis
    """
    box = []
    box2 = np.zeros(4)
    for b in bbox[2]:
        box.append(b)
    xc = box[0]
    yc = box[1]
    w  = box[2]
    h  = box[3]
    box2[0] = xc - w/2
    box2[1] = yc - h/2
    box2[2] = xc + w/2
    box2[3] = yc + h/2

    bbox[2] = box2
    return bbox


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    #print('Tracker: ',boxA)
    #print('Detector: ',boxB)

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    #print('IOU_STUFF:  ', interArea, boxAArea, boxBArea)
    # return the intersection over union value
    return iou


def detector(net, meta, image, thresh=0.3):
    detect = dn.detect(net, meta, image, thresh)
    '''print ('DETECTTTTTTTTTTTTT',detect)
    cv2.imshow("Cam2",image)
    cv2.waitKey(200)
    '''
    return detect

    
def object(detect):
    corners = []#[0]*4#, [0], [0], [0]]
    #print('DeTeCt',detect)
    try:
        for d in detect:
            #print(d)
            if d[0] == b'boat':
                corners = (yolo_boxes_to_corners(d))
    except:
        print('NONETYPE')
    return corners

def Horizon(image):
    #imgOg = cv2.imread(str(directory)+image) # Read image
    reduced = cv2.resize(image, (200, 200), interpolation = cv2.INTER_AREA)
    img_YUV = cv2.cvtColor(reduced,cv2.COLOR_BGR2YCR_CB)  # Convert from BGR to YCRCB
    b, r, g = cv2.split(img_YUV)                        # Split into blue-green-red channels
    #b = b*0
    #r = r*0
    #g = g*0
    imgBlueEqu = cv2.merge((cv2.equalizeHist(b), cv2.equalizeHist(r), cv2.equalizeHist(g))) # Equalize Blue Channel & Merge channels back
    img_BGR = cv2.cvtColor(imgBlueEqu,cv2.COLOR_YCR_CB2BGR)  # Convert from YCRCB to BGR

    blur = cv2.GaussianBlur(img_BGR,(9,9),0)            # Perform a gaussian blur
    b, r, g = cv2.split(blur)                        # Split into blue-green-red channels
    #b = b*0
    r = r*0
    g = g*0
    blur2 = cv2.merge((cv2.equalizeHist(b), r, g)) # Equalize Blue Channel & Merge channels back
    img_GREY = cv2.cvtColor(blur2,cv2.COLOR_BGR2GRAY)    # Convert to Greyscale

    ret, thresh = cv2.threshold(b,60,100,cv2.THRESH_BINARY) # Threshold image to segment sky

    # Perform erosian and dialiation to isolate the sky
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5), (3,3))
    img_erode = cv2.erode(thresh, kernel)
    img_dilate = cv2.dilate(img_erode, kernel)

    img, contours, hierarchy = cv2.findContours(img_dilate,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    cv2.imshow('Cam2', thresh)
    cv2.imshow('Cam3', blur2)
    if len(contour_sizes) > 0:
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        #print "\t"+str(image)+"\t"+str(cv2.contourArea(biggest_contour))+" \t\tPASS"
        #if output != 0:
        #final = cv2.drawContours(imgOg, biggest_contour, -1, (0,255,0), 3) # Outline elements that may be the sky
            #cv2.imwrite(output+"H_"+image,final)               # Write to file
        
        return biggest_contour
    #else:
        #print "\t"+str(image)+"\t\t\tFAIL"


# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Show the Webcam demo.')
    parser.add_argument('-r', '--record', action='store_true', default=False)
    args = parser.parse_args()
    RECORD = args.record

    #cam = cv2.VideoCapture('/home/runar/re3-tensorflow/demo/Bar.mp4')
    cam = cv2.VideoCapture('/home/runar/Ladybug/output0.mp4')
    cam.set(1, 6200-1)
    cv2.namedWindow('Cam', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('Cam2', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('Cam3', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('Cam4', cv2.WINDOW_NORMAL)
    ret_val, img = cam.read()

    #img = cv2.imread('left12.jpg')
    h, w = img.shape[:2]
    
    #h = int(h)
    #w = int(w)
    #M = h//3
    #N = w//2

    #mtx = np.matrix('1350.41716 0.0 1038.58110; 0.0 1352.74467 1219.10680; 0.0 0.0 1.0')
    mtx = np.matrix('1350.0 0.0 1024.0; 0.0 1350.0 1232.0; 0.0 0.0 1.0')
    #distort = np.array([-0.293594324, 0.0924910801, -0.000795067830, 0.000154218667, -0.0129375553])
    distort = np.array([-0.29, 0.09, -0.0, 0.0, -0.013])

    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,distort,(w,h),1,(w,h))
    # undistort
    #dst = cv2.undistort(img, mtx, distort, None, newcameramtx)

    # crop the image
    cropx, cropy = [216, 600]
    #img = dst[y:y+h, x:x+w]
    
    tracker = re3_tracker.Re3Tracker()

    #detector = "./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights /media/runar/Seagate\ Expansion\ Drive/Ladybug/1/output.mp4 -i 0"
    net = dn.load_net(b"/home/runar/yolov3.cfg", b"/home/runar/yolov3.weights", 0)
    meta = dn.load_meta(b"/home/runar/coco.data")
    #img = cv2.imread('/home/runar/boats.jpg')
    frameNum = 0
    #scale = 1
    #delta = 0
    #ddepth = cv2.CV_16S
    #horizon = []
    

    #while True:
    while not rospy.is_shutdown():
        rospy.spin()
        tiles = []
        corners = []
        #lines2 = []
        ret_val, img = cam.read()
        #cv2.imshow('Cam', img)
        
        # crop the image
        #x,y,w,h = roi
            
        
        if img is None:
            # End of video.
            break
        else:
            dst = cv2.undistort(img, mtx, distort, None, newcameramtx)
            h, w = dst.shape[:2]
            dimg = dst[cropy:h-cropy, cropx:w-cropx]
            #print("IMGSIZE",ret_val,cropx,cropy,w,h, roi, newcameramtx)
            h, w = dimg.shape[:2]
            im = dimg.copy()
            #cv2.imshow('Cam3', dst)
            #cv2.imshow('Cam4', dimg)

 
        
        if frameNum % 10 == 0:
            objects = []
            i = 0

            ## Detect larger objects covering most of the image
            #p1=detector(net, meta, dimg)
            #for o in p1:
            #    objects.append(o)

            ## Detect smaller objects
            tiles.append(dimg[h//3:(h//3)*2,0:(w//3)])
            tiles.append(dimg[h//3:(h//3)*2,(w//3):(w//3)*2])
            tiles.append(dimg[h//3:(h//3)*2,(w//3)*2:w])
            
            for tile in tiles:
                #cv2.imshow('Tile', tile)
                #cv2.waitKey(2000)
                p1p=detector(net, meta, tile)
                for d in p1p:
                    #print (d)
                    xc, yc, w1, h1 = d[2]
                    lst = list(d)
                    #print(lst)
                    lst[2] = ((xc+((w//3)*i)),(yc+((h//3))),w1,h1)
                    #print(lst)
                    objects.append(lst)
                i += 1

            corners.append(object(objects))
            #print("CORNERS",corners)
            if corners != [[]]:
                corners.sort(key=lambda x:x[1])
                corner = corners[0][2]
                show_webcam(dimg, corner)
            else:
                show_webcam(dimg)
                #corner = np.zeros(4)
            
            #print(corner)
            #show_webcam(img, corner)

        else:
            show_webcam(dimg)


        keyPressed = cv2.waitKey(1)
        if keyPressed == 27 or keyPressed == 1048603:
            break  # esc to quit
        elif keyPressed != -1:
            paused = True
        frameNum += 1
        
    cv2.destroyAllWindows()