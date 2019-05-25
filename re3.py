#!/usr/bin/env python
import cv2
import argparse
#import glob
#from skimage import transform as sk_transform
import numpy as np
#from scipy.spatial.transform import Rotation as R
import rospy
import os
import time
import sys
import math
import struct
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime
#from vidstab import VidStab


basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from re3.tracker import re3_tracker 
#from re3.tracker.darknet import darknet_orig as dn

from re3.re3_utils.util import drawing
from re3.re3_utils.util import bb_util
from re3.re3_utils.util import im_util

from re3.constants import OUTPUT_WIDTH
from re3.constants import OUTPUT_HEIGHT
from re3.constants import PADDING

#import autosea_msgs.msg as automsg
import std_msgs.msg as stdmsg
#import geometry_msgs.msg as geomsg
#import autoseapy.conversion as conv
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image
#from autoseapy.local_state import munkholmen
#import autoseapy.definitions as defs
#from sensor_msgs.msg import CameraInfo
#import darknet_ros_msgs.msg as darknetmsg
#from darknet_ros_msgs.msg import CheckForObjectsAction, CheckForObjectsGoal

#stabilizer = VidStab()
boxToDraw = None#np.zeros(4)
initialize = True

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
   
    return iou

class Re3Tracker(object):
    def __init__(self):
        # Params
        self.newimage = False
        self.corner = None
        self.penalty = 0
        self.time_past = 0
        #self.previmg = None
        #self.prev = None



    def show_webcam(self, image, corners=None):
        #print(corners)
        #self.image3 = self.warp.copy()
        h = self.height
        w = self.width
        global initialize, boxToDraw, tracker
        #if boxToDraw is None:
            #boxToDraw = corners
        if corners is not None:# and boxToDraw is not None:
            if boxToDraw is None:# and all(corners) > 0:
                boxToDraw = corners
                initialize = True
            iou = bb_intersection_over_union(boxToDraw,corners)
            if iou == 0:
                initialize = True
                print("New track")
            elif iou < 0.4:
                initialize = True
                print("Updated track") 
            if initialize:
                boxToDraw = corners
                initialize = False
                boxToDraw = tracker.track(image[:,:,::-1], 'Cam', boxToDraw)
                print(boxToDraw)
            else:
                boxToDraw = tracker.track(image[:,:,::-1], 'Cam')
        else:
            try:
                self.penalty += 1
                boxToDraw = tracker.track(image[:,:,::-1], 'Cam')
            except:
                print("No Bbox to track")
        if self.penalty > self.iterations:
            boxToDraw = None
        else:
            if boxToDraw is not None:
                if ((abs(boxToDraw[0]-boxToDraw[2]) > 4) and (abs(boxToDraw[1]-boxToDraw[3]) > 4)):
                    cv2.rectangle(self.image, (int(boxToDraw[0]), int(boxToDraw[1])), (int(boxToDraw[2]), int(boxToDraw[3])), 
                        [0,0,255], 2)
                    #self.bb_angle = self.fov_pixel*(int(boxToDraw[0])+(int(boxToDraw[2])-int(boxToDraw[0]))/2-self.width/2)+self.number*np.deg2rad(self.Mounting_angle)
                    a = boxToDraw
                    b = stdmsg.Float32MultiArray(data=a)
                    self.bb_publisher.publish(b)
                    #return boxToDraw
                else:
                    boxToDraw = None


    def bb_callback(self, msg):
        #print(msg.data)
        #self.newBB = True
        corner = msg.data
        if corner[2]-corner[0] > 5:
            self.penalty = 0
            self.corner = corner
        elif self.penalty > self.iterations/2 or boxToDraw is None:
            self.corner = corner

    def im_callback(self, msg):
        bridge = CvBridge()
        self.image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        #self.image = bridge.imgmsg_to_cv2(msg, "bgr8")
        '''
        grayimage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        if self.previmg is not None:
            self.prev = cv2.goodFeaturesToTrack(grayimage,mask = None,maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
            #corners = np.int0(corners)
            _nxt, status, error = cv2.calcOpticalFlowPyrLK(self.previmg, grayimage, self.prev, None, winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            # Selects good feature points for next position
            good_new = _nxt[status == 1]
            # Selects good feature points for previous position
            #if self.prev is not None:
            good_old = self.prev[status == 1]
            #else:
            #    good_old = good_new   
            # Updates previous frame
            # prev_gray = gray.copy()
            # Updates previous good feature points
            self.prev = good_new.reshape(-1, 1, 2)
            
            #_points, status, err = cv2.calcOpticalFlowPyrLK(self.previmg, grayimage, corners, np.array([]))
            # Filter out valid points only
            #points = _points[np.nonzero(status)]
            #corners = corners[np.nonzero(status)]
 
            h, w = grayimage.shape[:]
            #h0,w0 = self.previmg.shape[:]
            #hmin = min(h,h0)
            #wmin = min(w,w0)
            #greyprevimg = self.previmg[0:hmin,0:wmin]
            #greyimage = self.image[0:hmin,0:wmin]
            #print(good_old, good_new)
            mat, _ = cv2.estimateAffinePartial2D(good_old, good_new)
            #mat = cv2.estimateRigidTransform(good_old, good_new, 0)
            #print(test,mat)
            self.image = cv2.warpAffine(self.image,mat,(w,h))#,INTER_NEAREST|WARP_INVERSE_MAP)
            #self.image = cv2.warpPerspective(self.image,mat,(w,h))
            #self.window = cv2.findTransformECC(self.previmg, self.image, mat)
        self.previmg = grayimage
        '''
        self.height, self.width = self.image.shape[:2]
        self.time_past = time.time()
        self.newimage = True

    def start(self, number, length, padding=0):
        
        self.iterations = length
        #self.padding = padding
        # Subscribers
        #rospy.Subscriber('/seapath/pose',geomsg.PoseStamped, self.pose_callback)
        #rospy.Subscriber('/radar/estimates', automsg.RadarEstimate, self.radar_callback)
        #rospy.Subscriber('/ladybug/camera0/image_raw', Image, self.image_callback)
        #rospy.Subscriber('/darknet_ros/check_for_objects/result',darknetmsg.CheckForObjectsActionResult, self.darknet_callback)
        rospy.Subscriber(('/re3/bbox_new%s'%number), stdmsg.Float32MultiArray , self.bb_callback)
        rospy.Subscriber(('/re3/image%s'%number), Image , self.im_callback)

        self.bb_publisher = rospy.Publisher('/re3/bbox%s'%number, stdmsg.Float32MultiArray, queue_size=1)
        
        print('Camera:', number)
        while not rospy.is_shutdown():
            if self.newimage == True:
                self.newimage = False
                if time.time() - self.time_past < 5: 
                    if self.corner is not None:
                        p = [-padding, -padding, padding, padding]
                        self.corner = np.add(self.corner, p)
                        self.show_webcam(self.image, self.corner)
                    else:
                        self.show_webcam(self.image)
                    self.corner = None
                    try:
                        cv2.imshow(('Cam%s' %number), self.image)
                        cv2.waitKey(1)
                    except:
                        print('No image')
                else:
                    boxToDraw = None

# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", "-n", help="set camera number")
    parser.add_argument("--length", "-l", help="set tracking history length")
    parser.add_argument("--padding", "-p", help="set padding of bounding box")
    args = parser.parse_args()
    if args.number:
        number = int(args.number)  
        print("set parameters to %s" % args.number)
    else:
        number = 0
    if args.length:
        length = int(args.length)  
        print("set parameters to %s" % args.length)
    if args.padding:
        padding = int(args.padding)  
        print("set parameters to %s" % args.padding)

    #cv2.namedWindow('Cam', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Cam%s'%number, cv2.WINDOW_NORMAL)
    #cv2.namedWindow('Cam3', cv2.WINDOW_NORMAL) 
    
    tracker = re3_tracker.Re3Tracker()
    rospy.init_node("Re3Tracker", anonymous=True)
    # Setup Telemetron ownship tracker
    #telemetron_tf = TransformListener()

    DetectObjects_node = Re3Tracker()   
    DetectObjects_node.start(number, length, padding)