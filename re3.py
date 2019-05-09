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
        self.pose_dict = {}
        self.pose_arr = []
        self.newimage = False
        self.corner = None
        self.penalty = 0

    def pose_callback(self, msg):
        self.pose_stamp = float(str(msg.header.stamp))/1e9
        quat = msg.pose.orientation
        q_b_w = np.array([quat.x, quat.y, quat.z, quat.w])
        self.euler_angles = conv.quaternion_to_euler_angles(q_b_w)
        self.pose_dict[self.pose_stamp] = self.euler_angles
        self.pose_arr.append(self.pose_stamp)
        if len(self.pose_dict) > 1000:
            p = self.pose_arr.pop()
            del self.pose_dict[p]

    def darknet_callback(self, msg):
        dark_stamp = float(str(msg.header.stamp))/1e9
        dark_pose = self.pose_dict[dark_stamp]
        print('dark pose',dark_pose)
        ident = str(msg.result.id)
        _,ident2,_ = (str(msg.status.goal_id.id)).split('-')
        status = msg.status.status


        corners = None
        #corner = []
        h = self.height
        w = self.width
        #i = (int(ident2)+1)%4
        #i = (self.index)
        i = ident
        print('iiident: ',ident, self.index)
        
        dark_boxes = msg.result.bounding_boxes.bounding_boxes
        
        if dark_boxes != []:
            for d in dark_boxes:
                #print ('DARKER',d)
                dd = (str(d).splitlines())
                _,obj  = str(dd[0]).split(': ')
                _,prob = str(dd[1]).split(': ')
                _,xmin = str(dd[2]).split(': ')
                _,ymin = str(dd[3]).split(': ')
                _,xmax = str(dd[4]).split(': ')
                _,ymax = str(dd[5]).split(': ')
                #obj = obj[1] = obj.split('"')
                prob = float(prob)
                print('Prob: ', prob)

                if i < 3:
                    xmin = float(xmin) + ((w//3)*i)
                    ymin = float(ymin) + h//3
                    xmax = float(xmax) + ((w//3)*i)
                    ymax = float(ymax) + h//3
                else:
                    xmin = float(xmin)*3
                    ymin = float(ymin)*3*(3/4)
                    xmax = float(xmax)*3
                    ymax = float(ymax)*3*(3/4)

                if obj == '"boat"' and prob > 0.25:
                    if corners is None:
                        corners = []
                    corners.append([obj,prob,np.array([xmin,ymin,xmax,ymax])]) 

            if corners is not None:
                corners.sort(reverse = True, key=lambda x :x[1])
                corner = np.array(corners[0][2])
                cornerpose = self.rotation_3D(self.dark_pose[0]-self.euler_angles[0], self.dark_pose[1]-self.euler_angles[1], self.dark_pose[2]-self.euler_angles[2])
                A1 = np.array([ [1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0]])
                A2 = np.array([ [1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1],
                                [0, 0, 0]])
                pose = np.dot(np.dot(A1,cornerpose),A2)
                print(pose)
                cxymin = self.pixel_2_angle(corner[0], corner[1], 1)
                cxymax = self.pixel_2_angle(corner[2], corner[3], 1)
                rmin = np.dot(pose,cxymin)
                rmax = np.dot(pose,cxymax)
                xymin = self.angle_2_pixel(rmin[0],rmin[1],rmin[2])
                xymax = self.angle_2_pixel(rmax[0],rmax[1],rmax[2])
                #print(rmin,rmax,cxymin,cxymax)
                self.corner = [xymin[0],xymin[1], xymax[0],xymax[1]]
                #print(corner)
                #print('Detected boat: ',corner)
                #self.show_webcam(corner)
            
        if status == 3:
            #self.count += 1
            self.detect_ready = True
            self.darknet = int(ident2)

        else:
            print('YOLO ERROR STATUS')
            self.detect_ready = True

    def image_callback(self, msg):
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w = img.shape[:2]
        #self.focal = 1350
        #mtx = np.matrix('1350.41716 0.0 1038.58110; 0.0 1352.74467 1219.10680; 0.0 0.0 1.0')
        self.mtx = np.matrix('1350.0 0.0 1024.0; 0.0 1350.0 1232.0; 0.0 0.0 1.0')
        #distort = np.array([-0.293594324, 0.0924910801, -0.000795067830, 0.000154218667, -0.0129375553])
        self.distort = np.array([-0.29, 0.09, -0.0, 0.0, -0.013])

        self.newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.mtx,self.distort,(w,h),1,(w,h))

        # crop the image
        cropx, cropy = [216, 600]

        dst = cv2.undistort(img, self.mtx, self.distort, None, self.newcameramtx)
        h, w = dst.shape[:2]
        self.image = dst[cropy:h-cropy, cropx:w-cropx]
        self.height, self.width = self.image.shape[:2]
        self.newimage = True
        print(self.euler_angles)

    def show_webcam(self, image, corners=None):
        #self.image3 = self.warp.copy()
        h = self.height
        w = self.width
        global initialize, boxToDraw#,tracker
        #if boxToDraw is None:
            #boxToDraw = corners
        if corners is not None:# and boxToDraw is not None:
            if boxToDraw is None:# and all(corners) > 0:
                boxToDraw = corners
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
                self.penalty = 0
                print(boxToDraw)
            else:
                boxToDraw = tracker.track(image[:,:,::-1], 'Cam')
        else:
            try:
                self.penalty += 1
                boxToDraw = tracker.track(image[:,:,::-1], 'Cam')
            except:
                print("No Bbox to track")
        if self.penalty > 50:
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

    
    def rotation_3D(self, phi, theta, psi):
        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([ [1, 0, 0, 0],
                        [0, np.cos(phi), -np.sin(phi), 0],
                        [0, np.sin(phi), np.cos(phi), 0],
                        [0, 0, 0, 1]])
        
        RY = np.array([ [np.cos(theta), 0, -np.sin(theta), 0],
                        [0, 1, 0, 0],
                        [np.sin(theta), 0, np.cos(theta), 0],
                        [0, 0, 0, 1]])
        
        RZ = np.array([ [np.cos(psi), -np.sin(psi), 0, 0],
                        [np.sin(psi), np.cos(psi), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        return np.dot(np.dot(RX, RY), RZ)
        #print(R,RX,RY,RZ)

    def pixel_2_angle(self, x, y, z=1):
        x_angle = x*self.fov_pixel-(self.width/2)*self.fov_pixel+np.deg2rad(self.Mounting_angle)
        y_angle = y*self.fov_pixel-(self.height/2)*self.fov_pixel
        z_angle = z
        return [x_angle, y_angle, x_angle] 

    def angle_2_pixel(self, rx, ry, rz=0):
        x = (rx+(self.width/2)*self.fov_pixel-np.deg2rad(self.Mounting_angle))/self.fov_pixel
        y = (ry+(self.height/2)*self.fov_pixel)/self.fov_pixel
        z = 1
        return [x, y, z]

    def bb_callback(self, msg):
        #print(msg)
        #self.newBB = True
        self.corner = msg.data

    def im_callback(self, msg):
        bridge = CvBridge()
        self.image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        #self.image = bridge.imgmsg_to_cv2(msg, "bgr8")
        self.height, self.width = self.image.shape[:2]
        self.newimage = True

    def start(self):
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
                if self.corner is not None:
                    self.show_webcam(self.image, self.corner)
                else:
                    self.show_webcam(self.image)
                self.corner = None
                cv2.imshow('Cam2', self.image)
                cv2.waitKey(1)

# Main function
if __name__ == '__main__':
    #cv2.namedWindow('Cam', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Cam2', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('Cam3', cv2.WINDOW_NORMAL) 
    
    tracker = re3_tracker.Re3Tracker()
    rospy.init_node("Re3Tracker")
    # Setup Telemetron ownship tracker
    #telemetron_tf = TransformListener()

    DetectObjects_node = Re3Tracker()   
    DetectObjects_node.start()