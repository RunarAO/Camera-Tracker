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
        self.cam = 0
        self.old = None

        # Node cycle rate (in Hz).
        self.rate = rospy.Rate(10)
        rospy.Subscriber('/seapath/pose',geomsg.PoseStamped, self.pose_callback)
        rospy.Subscriber('/re3/camera', stdmsg.Int8, self.camera_callback)
        self.pub_image0 = rospy.Publisher('/ladybug/camera0/image_raw', Image, queue_size=1)
        self.pub_image1 = rospy.Publisher('/ladybug/camera1/image_raw', Image, queue_size=1)
        self.pub_image2 = rospy.Publisher('/ladybug/camera2/image_raw', Image, queue_size=1)
        self.pub_image3 = rospy.Publisher('/ladybug/camera3/image_raw', Image, queue_size=1)
        self.pub_image4 = rospy.Publisher('/ladybug/camera4/image_raw', Image, queue_size=1)

    def camera_callback(self, msg):
        if msg.data != self.cam:
            self.old = self.cam
            self.cam = msg.data
            print(self.cam, self.old)


    def pose_callback(self, msg):
        self.pose_stamp = float(str(msg.header.stamp))/1e9
        #quat = msg.pose.orientation
        #q_b_w = np.array([quat.x, quat.y, quat.z, quat.w])
        #self.euler_angles = conv.quaternion_to_euler_angles(q_b_w)


    def imageNametoTimestamp(self, stamp):
        #ladybug_18408820_20180927_124342_ColorProcessed_000699_Cam3_160568_024-3158.jpg
        #print(stamp[48:54])
        offset = 0.
        if self.firstimage == None:
            day = int(stamp[23:25])        ## Issues arrives at midnight
            month = int(stamp[21:23])
            year = int(stamp[17:21])
            #self.firstimage = int(stamp[48:54])
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
        milli2 = int(stamp[72:75])
        #milli = int(stamp[48:54])
        self.imagetimestamp = self.firstimagetimestamp + (milli+(milli2/1000)-self.firstimage)/10. + offset
        if self.newimagetimestamp > self.imagetimestamp:
            self.imagetimestamp = self.newimagetimestamp
            print('Timestamp updated')

    def start(self, number):
        bridge = CvBridge()
        sorted_file_list = {}
        video = {}
        i_set = {}
        #self.number = 0
        #self.Mounting_angle = 72       # 5 cameras, 360/5=72
        for cam in range(5): 
            im_dir = ("/home/runar/Skrivebord/%s" %cam)
            file_list = os.listdir(im_dir)
            sorted_file_list[cam] = sorted(file_list)#, key=lambda x:x[-30:])
            #video[cam] = cv2.VideoCapture('/home/runar/Ladybug/output%s.mp4' %cam)
            #i_set[cam] = False
        
        i = 1#4300
        
        #self.cam.set(1, 17000)
  
        self.image_time = str(sorted_file_list[0][i])
        self.imageNametoTimestamp(self.image_time)
 
        while not rospy.is_shutdown():
            if (self.imagetimestamp - self.pose_stamp) < -2:
                i += int(abs(self.imagetimestamp - self.pose_stamp))
                self.image_time = str(sorted_file_list[0][i])
                print("WAY SMALLER",self.imagetimestamp-self.pose_stamp, i) 
                self.imageNametoTimestamp(self.image_time)
            elif (self.imagetimestamp - self.pose_stamp) < -0.06:
                i += 2# + int(abs(self.imagetimestamp - self.pose_stamp))
                self.image_time = str(sorted_file_list[0][i])
                print("SMALLER",self.imagetimestamp-self.pose_stamp, i) 
                self.imageNametoTimestamp(self.image_time)
            elif (self.imagetimestamp - self.pose_stamp) > 0.06:
                if (self.imagetimestamp - self.pose_stamp) > 10:
                    i = 1       # Restart due to bag file restarted
                    print('LARGER',self.imagetimestamp-self.pose_stamp, i)
                    self.image_time = str(sorted_file_list[0][i])
                    self.imageNametoTimestamp(self.image_time)
            else:
                i += 1# + int(abs(self.imagetimestamp - self.pose_stamp))
                self.image_time = str(sorted_file_list[0][i])
                #print("GOOD",self.imagetimestamp-self.pose_stamp, i) 
                #print(self.euler_angles)
                self.imageNametoTimestamp(self.image_time)

            #for cam in range(5):
            #imdir = ("/home/runar/Skrivebord/%s" %number)
            #print(imdir, str(sorted_file_list[0][i-50]))
            imdir = ("/home/runar/Skrivebord/%s" %self.cam)
            image = cv2.imread(imdir + '/' + str(sorted_file_list[self.cam][i-number])) #51 fits 11.50 bag
            #if abs(self.imagetimestamp-self.pose_stamp) < 0.06 and i_set[cam] == False:
            #    video[cam].set(1, i-number)
            #    i_set[cam] = True
            #_,image = video[cam].read()
            im = bridge.cv2_to_imgmsg(image, 'bgr8')#encoding="passthrough")
            #if self.old is not None:# != self.old:
            '''
            imdir = ("/home/runar/Skrivebord/0")
            image = cv2.imread(imdir + '/' + str(sorted_file_list[0][i-number])) #51 fits 11.50 bag
            im0 = bridge.cv2_to_imgmsg(image, 'bgr8')#encoding="passthrough")
            imdir = ("/home/runar/Skrivebord/3")
            image = cv2.imread(imdir + '/' + str(sorted_file_list[3][i-number])) #51 fits 11.50 bag
            im3 = bridge.cv2_to_imgmsg(image, 'bgr8')#encoding="passthrough")
            imdir = ("/home/runar/Skrivebord/4")
            image = cv2.imread(imdir + '/' + str(sorted_file_list[4][i-number])) #51 fits 11.50 bag
            im4 = bridge.cv2_to_imgmsg(image, 'bgr8')#encoding="passthrough")

            self.pub_image0.publish(im0)
            self.pub_image3.publish(im3)
            self.pub_image4.publish(im4)
            '''
            #try:
            if self.cam == 0:
                #image = cv2.imread(imdir + '/' + str(sorted_file_list[cam][i-number])) #51 fits 11.50 bag
                #im = bridge.cv2_to_imgmsg(image, 'bgr8')#encoding="passthrough")
                self.pub_image0.publish(im)
            if self.cam == 1:
                self.pub_image1.publish(im)
            if self.cam == 2:
                self.pub_image2.publish(im)
            if self.cam == 3:
                #image = cv2.imread(imdir + '/' + str(sorted_file_list[cam][i-number])) #51 fits 11.50 bag
                #im = bridge.cv2_to_imgmsg(image, 'bgr8')#encoding="passthrough")
                self.pub_image3.publish(im)
            if self.cam == 4:
                #image = cv2.imread(imdir + '/' + str(sorted_file_list[cam][i-number])) #51 fits 11.50 bag
                #im = bridge.cv2_to_imgmsg(image, 'bgr8')#encoding="passthrough")
                self.pub_image4.publish(im)
            
            if self.old is not None and self.old != self.cam:
                print('changed camera')
                imdir = ("/home/runar/Skrivebord/%s" %self.old)
                image = cv2.imread(imdir + '/' + str(sorted_file_list[self.old][i-number])) #51 fits 11.50 bag
                im2 = bridge.cv2_to_imgmsg(image, 'bgr8')#encoding="passthrough")
                if self.old == 0:
                    self.pub_image0.publish(im2)
                if self.old == 1:
                    self.pub_image1.publish(im2)
                if self.old == 2:
                    self.pub_image2.publish(im2)
                if self.old == 3:
                    self.pub_image3.publish(im2)
                if self.old == 4:
                    self.pub_image4.publish(im2)
            self.old = None
            #except:
            #    print('No such image')
               
            self.rate.sleep()

# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", "-n", help="set camera number")
    args = parser.parse_args()
    if args.number:
        number = int(args.number)  
        print("set camera number to %s" % args.number)
    #cv2.namedWindow('Cam', cv2.WINDOW_NORMAL)
    rospy.init_node("CameraImage")
    DetectObjects_node = SendImage()   
    DetectObjects_node.start(number)