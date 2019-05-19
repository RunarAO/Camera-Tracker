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
    print('IOU')
    return iou
'''
class Re3Tracker(object):
    def __init__():
        # Params
        self.corner = None
        self.penalty = 0
        
        #self.previmg = None
        #self.prev = None



    def show_webcam(self, image, corners=None):
        #print(corners)
        #self.image3 = self.warp.copy()
        #h = self.height
        #w = self.width
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
                else:
                    boxToDraw = None
        return boxToDraw

    def start(self, number, padding=0):
        self.iterations = number
        #self.padding = padding
        # Subscribers
        #rospy.Subscriber('/seapath/pose',geomsg.PoseStamped, self.pose_callback)
        #rospy.Subscriber('/radar/estimates', automsg.RadarEstimate, self.radar_callback)
        #rospy.Subscriber('/ladybug/camera0/image_raw', Image, self.image_callback)
        #rospy.Subscriber('/darknet_ros/check_for_objects/result',darknetmsg.CheckForObjectsActionResult, self.darknet_callback)
        rospy.Subscriber('/re3/bbox_new', stdmsg.Float32MultiArray , self.bb_callback)
        rospy.Subscriber('/re3/image', Image , self.im_callback)

        self.bb_publisher = rospy.Publisher('/re3/bbox', stdmsg.Float32MultiArray, queue_size=1)

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
                        cv2.imshow('Cam2', self.image)
                        cv2.waitKey(1)
                    except:
                        print('No image')
                else:
                    boxToDraw = None
'''
class re3tracking(object):
    def __init__(self):
        self.tracker = {}
        self.image = {}
        self.corner = {}
        self.box = {}
        self.penalty = {}
        self.number = None
        self.newimage = {}
        self.time_past = {}

        rospy.Subscriber('/re3/bbox_new', stdmsg.Float32MultiArray , self.bb_callback)
        rospy.Subscriber('/re3/image', Image , self.im_callback)
        rospy.Subscriber('/re3/number', stdmsg.Int32, self.number_callback)
        self.bb_publisher = rospy.Publisher('/re3/bbox', stdmsg.Float32MultiArray, queue_size=1)


    def number_callback(self, msg):
        self.number = int(msg.data)

    def bb_callback(self, msg):
        #print(self.number, msg)
        if self.number is not None:
            corner = msg.data
            i = self.number
            if corner[2]-corner[0] > 5:
                self.penalty[i] = 0
                self.corner[i] = corner
            elif corner[3]-corner[1] == 0:
                self.corner[i] = corner

    def im_callback(self, msg):

        if self.number is not None:
            i = self.number
            initialize = False
            print('IMAGE',i)
            bridge = CvBridge()
            self.image[i] = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            #self.height, self.width = self.image.shape[:2]
            self.time_past[i] = time.time()
            #self.newimage[i] = True
            if i not in self.tracker:
                self.tracker[i] = re3_tracker.Re3Tracker()
                p = [-padding, -padding, padding, padding]
                corner = np.add(self.corner[i], p)
                self.box[i] = self.tracker[i].track(self.image[i][:,:,::-1], i, corner)
                self.penalty[i] = 0
                print('new',self.box)
            for j in self.tracker:
                #print(j,i)
                if j == i:
                    if time.time() - self.time_past[i] < 5: 
                        image = self.image[i][:,:,::-1]
                        #print(self.corner[i])
                        if i in self.corner:
                            print(self.corner[i])
                            if self.corner[i] is not None:
                            #if self.corner[i][3]-self.corner[i][1] == 0 and self.
                                if i not in self.box:
                                    self.box[i] = self.corner[i]
                                    initialize = True
                                elif self.box[i] is not None:
                                #else:
                                    iou = bb_intersection_over_union(self.box[i],self.corner[i])
                                    if iou < 0.5:
                                        initialize = True
                                if initialize:
                                    print('init')
                                    p = [-padding, -padding, padding, padding]
                                    corner = np.add(self.corner[i], p)
                                    self.box[i] = self.tracker[i].track(image, i, corner)
                                else:
                                    self.box[i] = self.tracker[i].track(image, i)
                        elif self.box[i] is not None:
                            self.box[i] = self.tracker[i].track(image, i)
                            self.penalty += 1
                        if self.penalty > self.iterations:
                            self.box[i] = None
                        print(self.box)
                        try:
                            b = self.box[i]
                            cv2.rectangle(self.image[i], (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), 
                                [0,0,255], 2)
                            cv2.imshow(str(i), self.image[i])
                            cv2.waitKey(1)
                            #a = self.box[i]
                            c = stdmsg.Float32MultiArray(data=[i,b])
                            self.bb_publisher.publish(c)
                        except:
                            print('No image')
                    else:
                        del self.tracker[i]
                        break
                #else:
                #    break
            
                        
                    self.corner[i] = None

        #self.re3[i] = Re3Tracker(self.image[i], self.corner[i])

    def start(self, iterations=10, padding=0):
        self.iterations = iterations
        self.newimage[0] = False
        
        while not rospy.is_shutdown():
            '''
            i = self.number
            initialize = False
            if self.newimage != {}:
                if self.newimage[i] == True:
                    self.newimage[i] = False
                    if i not in self.tracker:
                        self.tracker[i] = re3_tracker.Re3Tracker()
                        p = [-padding, -padding, padding, padding]
                        corner = np.add(self.corner[i], p)
                        self.box[i] = self.tracker[i].track(self.image[i][:,:,::-1], i, corner)
                        self.penalty[i] = 0
                        print('new',self.box)
                    for j in self.tracker:
                        #print(j,i)
                        if j == i:
                            if time.time() - self.time_past[i] < 5: 
                                image = self.image[i][:,:,::-1]
                                #print(self.corner[i])
                                if self.corner[i] is not None:
                                    #if self.corner[i][3]-self.corner[i][1] == 0 and self.
                                    if self.box[i] is None:
                                        self.box[i] = self.corner[i]
                                        initialize = True
                                    else:
                                        iou = bb_intersection_over_union(self.box[i],self.corner[i])
                                        if iou < 0.5:
                                            initialize = True
                                    if initialize:
                                        print('init')
                                        p = [-padding, -padding, padding, padding]
                                        corner = np.add(self.corner[i], p)
                                        self.box[i] = self.tracker[i].track(image, i, corner)
                                    else:
                                        self.box[i] = self.tracker[i].track(image, i)
                                elif self.box[i] is not None:
                                    self.box[i] = self.tracker[i].track(image, i)
                                    self.penalty += 1
                                if self.penalty > self.iterations:
                                    self.box[i] = None
                                print(self.box)
                                try:
                                    b = self.box[i]
                                    cv2.rectangle(self.image[i], (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), 
                                        [0,0,255], 2)
                                    cv2.imshow(str(i), self.image[i])
                                    cv2.waitKey(1)
                                    #a = self.box[i]
                                    c = stdmsg.Float32MultiArray(data=[i,b])
                                    self.bb_publisher.publish(c)
                                except:
                                    print('No image')
                            else:
                                del self.tracker[i]
                                break
                        else:
                            break
                    
                        
                self.corner[i] = None
            #except:
            #    print('image not ready')
            '''

# Main function
if __name__ == '__main__':
    number = 10
    padding = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", "-n", help="set tracking history")
    parser.add_argument("--padding", "-p", help="set padding of bounding box")
    args = parser.parse_args()
    if args.number:
        number = int(args.number)  
        print("set parameters to %s" % args.number)
    if args.padding:
        padding = int(args.padding)  
        print("set parameters to %s" % args.padding)

    #cv2.namedWindow('Cam', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Cam2', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('Cam3', cv2.WINDOW_NORMAL) 
    
    #tracker = re3_tracker.Re3Tracker()
    rospy.init_node("Re3Tracker")
    # Setup Telemetron ownship tracker
    #telemetron_tf = TransformListener()

    DetectObjects_node = re3tracking()   
    #DetectObjects_node = Re3Tracker()   
    DetectObjects_node.start(number, padding)