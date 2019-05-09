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

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from re3.tracker import re3_tracker 

from re3.re3_utils.util import drawing
from re3.re3_utils.util import bb_util
from re3.re3_utils.util import im_util

from re3.constants import OUTPUT_WIDTH
from re3.constants import OUTPUT_HEIGHT
from re3.constants import PADDING

import autosea_msgs.msg as automsg
import std_msgs.msg as stdmsg
import geometry_msgs.msg as geomsg
import autoseapy.conversion as conv
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image
from autoseapy.local_state import munkholmen
import autoseapy.definitions as defs


class SendImage(object):
    def __init__(self):
        # Params
        self.firstimage = None
        self.newimagetimestamp = 0
        self.pose_stamp = 0

        # Node cycle rate (in Hz).
        self.rate = rospy.Rate(10)
        rospy.Subscriber('/seapath/pose',geomsg.PoseStamped, self.pose_callback)
        self.pub_image = rospy.Publisher('/ladybug/camera0/image_raw', Image, queue_size=1)


    def pose_callback(self, msg):
        self.pose_stamp = float(str(msg.header.stamp))/1e9
        quat = msg.pose.orientation
        q_b_w = np.array([quat.x, quat.y, quat.z, quat.w])
        self.euler_angles = conv.quaternion_to_euler_angles(q_b_w)


    def imageNametoTimestamp(self, stamp):
        #ladybug_18408820_20180927_124342_ColorProcessed_000699_Cam3_160568_024-3158.jpg
        #print(stamp[23:25])
        offset = 0.
        if self.firstimage == None:
            day = int(stamp[23:25])        ## Issues arrives at midnight
            month = int(stamp[21:23])
            year = int(stamp[17:21])
            self.firstimage = int(stamp[60:66])
            hour = int(stamp[26:28])
            minute = int(stamp[28:30])
            second = int(stamp[30:32])
            imagetime = datetime(year, month, day, hour, minute, int(second), int((second%1)*1000))
            self.firstimagetimestamp = time.mktime(imagetime.timetuple())
        else:
            day = int(stamp[23:25])        ## Issues arrives at midnight
            month = int(stamp[21:23])
            year = int(stamp[17:21])
            hour = int(stamp[26:28])
            minute = int(stamp[28:30])
            second = int(stamp[30:32])
            imagetime = datetime(year, month, day, hour, minute, int(second), int((second%1)*1000))
            self.newimagetimestamp = time.mktime(imagetime.timetuple())

        milli = int(stamp[60:66])
        self.imagetimestamp = self.firstimagetimestamp + (milli-self.firstimage)/10. + offset
        if self.newimagetimestamp > self.imagetimestamp:
            self.imagetimestamp = self.newimagetimestamp
            print('Timestamp updated')

    def start(self, number):
        bridge = CvBridge()
        #self.number = 0
        #self.Mounting_angle = 72       # 5 cameras, 360/5=72

        im_dir = ("/home/runar/Skrivebord/%s" %number)
        file_list = os.listdir(im_dir)
        sorted_file_list = sorted(file_list)#, key=lambda x:x[-30:])
        i = 1#4300
        self.cam = cv2.VideoCapture('/home/runar/Ladybug/output0.mp4')
        #self.cam.set(1, 17000)
  
        self.image_time = str(sorted_file_list[i])
        self.imageNametoTimestamp(self.image_time)
 
        while not rospy.is_shutdown():
            if (self.imagetimestamp - self.pose_stamp) < -2:
                i += int(abs(self.imagetimestamp - self.pose_stamp))
                self.image_time = str(sorted_file_list[i])
                print("WAY SMALLER",self.imagetimestamp-self.pose_stamp, i) 
                self.imageNametoTimestamp(self.image_time)
            elif (self.imagetimestamp - self.pose_stamp) < -0.06:
                i += 2# + int(abs(self.imagetimestamp - self.pose_stamp))
                self.image_time = str(sorted_file_list[i])
                #print("SMALLER",self.imagetimestamp-self.pose_stamp, i) 
                self.imageNametoTimestamp(self.image_time)
            elif (self.imagetimestamp - self.pose_stamp) > 0.06:
                if (self.imagetimestamp - self.pose_stamp) > 10:
                    i = 1       # Restart due to bag file restarted
                    print('LARGER',self.imagetimestamp-self.pose_stamp, i)
                    self.image_time = str(sorted_file_list[i])
                    self.imageNametoTimestamp(self.image_time)
            else:
                i += 1# + int(abs(self.imagetimestamp - self.pose_stamp))
                self.image_time = str(sorted_file_list[i])
                #print("GOOD",self.imagetimestamp-self.pose_stamp, i) 
                #print(self.euler_angles)
                self.imageNametoTimestamp(self.image_time)

            image = cv2.imread(im_dir + '/' + sorted_file_list[i-50]) #51 fits 11.50 bag
            #ret_val, image = self.cam.read()
            #cv2.imshow('Cam',image)
            #cv2.waitKey(1)
            im = bridge.cv2_to_imgmsg(image, 'bgr8')#encoding="passthrough")
            self.pub_image.publish(im)
            #self.rate.sleep()

# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", "-n", help="set camera number")
    args = parser.parse_args()
    if args.number:
        number = args.number  
        print("set camera number to %s" % args.number)
    #cv2.namedWindow('Cam', cv2.WINDOW_NORMAL)
    rospy.init_node("CameraImage")
    DetectObjects_node = SendImage()   
    DetectObjects_node.start(number)