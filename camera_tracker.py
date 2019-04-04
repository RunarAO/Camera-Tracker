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
#import tf
#from tf import TransformListener
#import message_filters
from datetime import datetime
import matplotlib.pyplot as plt


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


import autosea_msgs.msg as automsg
import std_msgs.msg as stdmsg
import geometry_msgs.msg as geomsg
import autoseapy.conversion as conv
identity_quat = geomsg.Quaternion(0,0,0,1)
identity_pos = geomsg.Point(0,0,0)
identity_pose = geomsg.Pose(position=identity_pos, orientation=identity_quat)
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image
from autoseapy.local_state import munkholmen
import autoseapy.definitions as defs
#from sensor_msgs.msg import CameraInfo

import actionlib
import darknet_ros_msgs.msg as darknetmsg
from darknet_ros_msgs.msg import CheckForObjectsAction, CheckForObjectsGoal
#import darknet_ros_msgs.action as darknetaction


np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)


boxToDraw = None#np.zeros(4)
initialize = True



'''
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
'''

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

'''
def detector(image, net=0, meta=0, thresh=0.3):
    
    detect = dn.detect(net, meta, image, thresh)
    #detect = rospy.Publisher('image', darknetaction.CheckForObjects, queue_size=5)
    #detect = darknetmsg.BoundingBoxes([])
    #pub = rospy.Publisher('/ladybug/object_img/image_raw', Image, queue_size=15)
    #pub.publish(image)
    #detect = darknetmsg.BoundingBox()
    #detect = DetectObjects(image)
    #self._as = actionlib.SimpleActionServer(self._action_name, actionlib_tutorials.msg.FibonacciAction, execute_cb=self.execute_cb, auto_start = False)
    #self._as.start()
    print ('DETECTTTTTTTTTTTTT',detect)
    cv2.imshow("Cam2",image)
    cv2.waitKey(200)
    return detect

    
def findobject(detect):
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
'''
'''
def Horizon(self,image):
    lines = []
    #imgOg = cv2.imread(str(directory)+image) # Read image
    reduced = cv2.resize(image, (400, 400), interpolation = cv2.INTER_AREA)
    img_gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray,(9,9),0)            # Perform a gaussian blur
    edges = cv2.Canny(blur,50,150,apertureSize = 3)
    #lines = cv2.HoughLinesP(edges,1,np.pi/180,100,80,10)
    lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 100,minLineLength = 100,maxLineGap = 50)
    try:
    #for i in range(N):
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(reduced,(x1,y1),(x2,y2),(0,255,0),2)
            np.arctan2((x2-x1),(y2-y1))
    except:
        print('OHNO')
    cv2.imshow('Cam2',reduced)
    plt.plot_date(x=days, y=impressions, fmt="r-")
    
    #img_YUV = cv2.cvtColor(reduced,cv2.COLOR_BGR2YCR_CB)  # Convert from BGR to YCRCB
    #b, r, g = cv2.split(img_YUV)                        # Split into blue-green-red channels
    #b = b*0
    #r = r*0
    #g = g*0
    #imgBlueEqu = cv2.merge((cv2.equalizeHist(b), cv2.equalizeHist(r), cv2.equalizeHist(g))) # Equalize Blue Channel & Merge channels back
    #img_BGR = cv2.cvtColor(imgBlueEqu,cv2.COLOR_YCR_CB2BGR)  # Convert from YCRCB to BGR

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
    #cv2.imshow('Cam3', blur2)
    if len(contour_sizes) > 0:
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        #print "\t"+str(image)+"\t"+str(cv2.contourArea(biggest_contour))+" \t\tPASS"
        #if output != 0:
        #final = cv2.drawContours(imgOg, biggest_contour, -1, (0,255,0), 3) # Outline elements that may be the sky
            #cv2.imwrite(output+"H_"+image,final)               # Write to file
        
        return biggest_contour
    #else:
        #print "\t"+str(image)+"\t\t\tFAIL"
'''   
'''
def inverse_transform(pos, quat):
    pos_np = np.array([pos.x, pos.y, pos.z])
    q_np = conv.quaternion(quat.x, quat.y, quat.z, quat.w)
    (pos_inv, q_inv) = conv.quat_inverse_transform(pos_np, q_np)
    return pos_inv, q_inv

def publish_seapath_pose(msg, llh0=munkholmen):
    #global pos
    pos = msg.pose.position
    quat = msg.pose.orientation
    (pos_inv, q_inv) = inverse_transform(pos, quat)
    # coordinate transform from world to body
    broadcaster = tf.TransformBroadcaster()
    broadcaster.sendTransform(tuple(pos_inv), tuple(q_inv), msg.header.stamp, defs.world_seapath, defs.body_seapath)
    # coordinate transform from surface to body
    q_b_w = np.array([quat.x, quat.y, quat.z, quat.w])
    #global euler_angles 
    euler_angles = conv.quaternion_to_euler_angles(q_b_w)
    #return euler_angles
    #q_w_s = conv.euler_angles_to_quaternion(np.array([0,0,-euler_angles[2]]))
    #q_s_b = conv.quat_conj(conv.quat_mul(q_w_s, q_b_w))
    #broadcaster.sendTransform((0,0,0), tuple(q_s_b), msg.header.stamp, defs.surface_seapath, defs.body_seapath)
    #broadcaster.sendTransform((0,0,0), tuple(q_w_s), msg.header.stamp, defs.NED_seapath, defs.surface_seapath)
'''

class DetectObjects(object):
    def __init__(self):
        # Params
        self.darknet = None
        self.detect_ready = True
        self.count = 0
        self.index = 0
        #self.q_b_w = []
        self.pos = []
        #self.ned = []
        self.euler_angles = []
        #self.tile = []
        #self.tiles = []
        self.image = []
        #self.radar_img = []
        self.millisecond = None
        self.firstimage = None
        self.second = None
        self.minute = None
        self.hour = None
        self.newimagetimestamp = 0
        self.radar_pixel = 0
        self.detections = []
        self.radar_detections = []
        self.range = 1000
        self.bb_angle = 0
        #self.dimg = []
        #self.bridge = CvBridge()
        #with np.load('calib.npz') as X:
        #    self.mtx, self.dist, self.rvec, self.tvec = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
        #print(self.mtx, self.dist, self.rvec, self.tvec)

        # Node cycle rate (in Hz).
        self.rate = rospy.Rate(10)

        # Publishers
        #self.pub_tile = rospy.Publisher('/ladybug/object_img/image_raw', Image, queue_size=1)
        #self.pub_goal = rospy.Publisher('/darknet_ros/check_for_objects/goal', darknetmsg.CheckForObjectsActionGoal, queue_size=1)
        self.dark_client = actionlib.SimpleActionClient('darknet_ros/check_for_objects', darknetmsg.CheckForObjectsAction)
        #self.pub_calib = rospy.Publisher('/ladybug/calib_img/image_calib', Image, queue_size=5)

        # Subscribers
        rospy.Subscriber('/seapath/pose',geomsg.PoseStamped, self.pose_callback)
        #scan_topic = rospy.get_param('~scan_topic', 'radar_scans')
        #rospy.Subscriber('/radar/radar_scans', automsg.RadarScan, self.radar_callback)#, callback_args=(track_publisher, track_manager, telemetron_tf, measurement_covariance_parameters), queue_size=30)
        rospy.Subscriber('/radar/estimates', automsg.RadarEstimate, self.radar_callback)
        #rospy.Subscriber('/radar/clusters', automsg.RadarCluster, self.cluster_callback)
        #rospy.Subscriber('/mr/spokes', automsg.RadarSpoke, self.spoke_callback )
        #rospy.Subscriber('/darknet_ros/found_object', stdmsg.Int8, self.darknet2_callback)
        #rospy.Subscriber('/darknet_ros/bounding_boxes',darknetmsg.BoundingBoxes, self.darknet_callback)
        rospy.Subscriber('/darknet_ros/check_for_objects/result',darknetmsg.CheckForObjectsActionResult, self.darknet3_callback)
        #rospy.Subscriber('/darknet_ros/detection_image', Image, self.darknet_detect_image)

        #self.pose = message_filters.Subscriber('/seapath/pose',geomsg.PoseStamped)
        #rospy.Subscriber('/ladybug/camera'+str(self.number)+'/image_raw', Image, self.image_callback)
        #self.cam = cv2.VideoCapture('/home/runar/Ladybug/output0.mp4')
        #self.cam.set(1, 17000-1)
        #ret_val, img = self.cam.read()
        # try:
        #     img = self.bridge.imgmsg_to_cv2(rosimg, "bgr8")
        #     img = self.bridge.imgmsg_to_cv2(rosimg, desired_encoding="passthrough")
        # except CvBridgeError as e:
        #     print(e)


            
            #ts = message_filters.ApproximateTimeSynchronizer([self.pose, self.image], 10, 10, allow_headerless=True)
            #ts = message_filters.TimeSynchronizer([self.pose, self.image], 10)
            #ts.registerCallback(self.callback)


    def extended_kalman(self, msg):
        print('EKF: ',msg)



    def data_assosiation(self):
        print('DATA')
        a = []
        c = []
        #try:
        if self.radar_detections != []:
            for d in self.radar_detections:
                #print('A')
                a.append([abs(int(d[0])-self.bb_angle), abs(int(d[1])-self.range), self.bb_angle, d])
                #e[1] = d[1]
                #e[2] = self.bb_angle
                #e[3] = d
                #print('e:  ', e)
            self.radar_detections = []
            try:
                a.sort(key=lambda x:x[0])
            except:
                print('a sort failed')
            for b in a:
                #print('b')
                if b[0] < 0.1:
                    c.append(b)
                    #print('B')
            try:
                c.sort(key=lambda x:x[1])
                self.range = c[0][1]
                print('RANGE: ',self.range)
                self.extended_kalman(c[0][3])
            except:
                print('c failed')
        else:
            print('No radar detections')
        #except:
        #    print('Radar assosiation error')
        #print('NOE')

    def darknet_detect_image(self, msg):
        bridge = CvBridge()
        im = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        print(im.shape[:])
        cv2.imshow('test', im)
        cv2.waitKey(1)

    def darknet3_callback(self, msg):
        self.dark_stamp = float(str(msg.header.stamp))/1e9
        #self.ident = str(msg.result.id)
        _,ident2,_ = (str(msg.status.goal_id.id)).split('-')
        status = msg.status.status


        corners = None
        corner = []
        h, w = self.image.shape[:2]
        i = (int(ident2)+1)%4
        
        dark_boxes = msg.result.bounding_boxes.bounding_boxes
        
        if dark_boxes != []:
            for d in dark_boxes:
                #print ('DARKER',d)
                dd = (str(d).splitlines())
                _,obj = str(dd[0]).split(': ')
                _,prob = str(dd[1]).split(': ')
                _,xmin = str(dd[2]).split(': ')
                _,ymin = str(dd[3]).split(': ')
                _,xmax = str(dd[4]).split(': ')
                _,ymax = str(dd[5]).split(': ')
                #obj = obj[1] = obj.split('"')
                prob = float(prob)

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

                if obj == '"boat"':# and prob > 0.2:
                    if corners is None:
                        corners = []
                    corners.append([obj,prob,np.array([xmin,ymin,xmax,ymax])]) 

            if corners is not None:
                corners.sort(reverse = True, key=lambda x :x[1])
                corner = np.array(corners[0][2])
                #print('Detected boat: ',corner)
                self.show_webcam(self.image, corner)
            else:
                self.show_webcam(self.image)

        if status == 3:
            #self.count += 1
            self.detect_ready = True
            self.darknet = int(ident2)
        else:
            print('YOLO ERROR STATUS')
            self.detect_ready = True

    def darknet2_callback(self, msg):
        num = str(msg).split(': ')
        _,n = num
        if int(n) == 0:
            self.detect_ready = True
            #print(num, self.count)
        self.count += 1
        

    def darknet_callback(self, msg):
        self.dark_stamp = float(str(msg.header.stamp))/1e9
        self.dark_image = str(msg.image_header)
        self.dark_boxes = msg.bounding_boxes
        print('DARKER  ',self.index, self.count)
        #print('DARK   ',self.dark_stamp, self.dark_image, self.dark_boxes)
        corners = []
        corner = []
        h, w = self.image.shape[:2]
        i = self.index
        for d in self.dark_boxes:
            #print ('DARKER',d)
            dd = (str(d).splitlines())
            _,obj = str(dd[0]).split(': ')
            _,prob = str(dd[1]).split(': ')
            _,xmin = str(dd[2]).split(': ')
            _,ymin = str(dd[3]).split(': ')
            _,xmax = str(dd[4]).split(': ')
            _,ymax = str(dd[5]).split(': ')
            #obj = obj[1] = obj.split('"')
            prob = float(prob)
            xmin = float(xmin) + ((w//3)*i)
            ymin = float(ymin) + h//3
            xmax = float(xmax) + ((w//3)*i)
            ymax = float(ymax) + h//3
            #xmin = xmin.split(': ')
            #print(obj,prob,xmin, xmax, 'INDEX : ',self.index)
            if obj == '"boat"':# and prob > 0.2:
                corners.append([obj,prob,np.array([xmin,ymin,xmax,ymax])]) 
            #xc, yc, w1, h1 = d[2]
            #lst = list(d)
            #print(lst)
            #lst[2] = ((xc+((w//3)*i)),(yc+((h//3))),w1,h1)
            #print(lst)
            #objects.append(lst)

            #corners.append(findobject(objects))
                #print("CORNERS",corners)
        if corners is not []:
            corners.sort(key=lambda x :x[1])
            corner = np.array(corners[-1][2])
            print('CORNER',corner)
            self.show_webcam(self.image, corner)
                #else:
                #    self.show_webcam(self.image)
                #corner = np.zeros(4)
        self.detect_ready = True
        #self.count += 1
        
        

    def pose_callback(self, msg):
        self.pose_stamp = float(str(msg.header.stamp))/1e9
        #self.pose_time = datetime.fromtimestamp(self.pose_stamp)
        self.position = msg.pose.position
        quat = msg.pose.orientation
        #self.ned = np.array([self.position.x, self.position.y, self.position.z])
        q_b_w = np.array([quat.x, quat.y, quat.z, quat.w])
        self.euler_angles = conv.quaternion_to_euler_angles(q_b_w)
        #print('ANGLES', self.euler_angles)
        #print(self.mtx, self.dist, self.rvec, self.tvec)

    def radar_callback(self, msg):
        #print(msg)
        self.radar_stamp = float(str(msg.header.stamp))/1e9
        #print(self.radar_stamp, self.pose_stamp)
        est_track_id = msg.track_id
        posterior_pos = msg.posterior.pos_est
        posterior_vel = msg.posterior.vel_est
        posterior_pos_cov = msg.posterior.pos_cov
        posterior_vel_cov = msg.posterior.vel_cov
        #print(self.posterior_pos, self.posterior_vel)
        self.posterior_ned = np.array([posterior_pos.x, posterior_pos.y])
        #print(self.radar_stamp)
        #print(self.centroid_ned, self.ned)
        dx = posterior_pos.x - self.position.x
        dy = posterior_pos.y - self.position.y
        phi, theta, psi = self.euler_angles

        radar_angle_ned = np.arctan2(dx,dy)
        self.radar_angle_body = radar_angle_ned - psi + np.deg2rad(-3.14)     # Installation angle offset between camera and radar  

        if self.radar_angle_body < -np.pi:
            self.radar_angle_body += 2*np.pi
        elif self.radar_angle_body > np.pi:
            self.radar_angle_body -= 2*np.pi

        self.radar_angle_image = self.radar_angle_body - self.number*np.deg2rad(self.Mounting_angle)
        self.radar_pixel = int(self.radar_angle_image/self.fov_pixel)
        self.radar_range = np.sqrt(dx**2+dy**2)
        self.radar_detections.append(np.array([self.radar_angle_body, self.radar_range, 
            [posterior_vel.x, posterior_vel.y], [posterior_pos_cov.var_x, posterior_pos_cov.var_y, posterior_pos_cov.cor_xy], 
            [posterior_vel_cov.var_x, posterior_vel_cov.var_y, posterior_vel_cov.cor_xy]]))
        self.detections.append(self.radar_pixel)

        cv2.line(self.warp, (self.radar_pixel, 0), (self.radar_pixel, self.height), (0,255,0), 10)
        #print('RADAR_ANGLE',radar_angle_ned, psi, self.radar_angle_body, self.radar_angle_image, self.radar_pixel)

    def cluster_callback(self, msg):
        
        self.cluster_stamp = float(str(msg.header.stamp))/1e9
        #self.est_track_id = msg.track_id
        #print(self.est_track_id)
        self.centroid_pos = msg.centroid
        #self.posterior_vel = msg.posterior.vel_est
        #print(self.posterior_pos, self.posterior_vel)
        self.centroid_ned = np.array([self.centroid_pos.x, self.centroid_pos.y])
        #print(self.radar_stamp)
        #print(self.centroid_ned, self.ned)
        dx = self.centroid_pos.x - self.position.x
        dy = self.centroid_pos.y - self.position.y
        phi, theta, psi = self.euler_angles

        cluster_angle_ned = np.arctan2(dx,dy)
        self.cluster_angle_body = cluster_angle_ned - psi + np.deg2rad(3)     # Installation angle offset between camera and radar  
        if self.cluster_angle_body < -np.pi:
            self.cluster_angle_body += 2*np.pi
        elif self.cluster_angle_body > np.pi:
            self.cluster_angle_body -= 2*np.pi
        self.cluster_pixel = int(self.cluster_angle_body/self.fov_pixel)
        #if abs(self.radar_pixel) < (self.width/2):
        self.cdetections.append(self.cluster_pixel)
            #self.radar_img = self.warp.copy()
            #cv2.line(self.warp, (self.radar_pixel, 0), (self.radar_pixel, self.height), (0,255,0), 10)
            #cv2.waitKey(100)
        #print('RADAR_ANGLE',radar_angle_ned, psi, self.radar_angle_body, self.radar_pixel, self.width)
        self.cluster_range = np.sqrt(dx**2+dy**2)
        #print(self.radar_range)
        #print(msg)
        #cv2.imshow('Cam3', self.warp)

    def spoke_callback(self, msg):
        self.spoke_stamp = float(str(msg.header.stamp))/1e9
        self.azimuth = msg.azimuth
        self.intensity = msg.intensity
        for i in self.intensity:
            d = struct.unpack("c", i)
            if d != "\x00":
                print(d)
            with open('anotherfile.txt', 'a') as the_file:
                the_file.write(str(i)+'\n')
        with open('anotherfile.txt', 'a') as the_file:
                the_file.write('THE__END '+'\n') 
            #d = int(i,32)
            #print(str(d))
            #for ii in i:
            #    print(self.intensity)
            #print(self.azimuth)

        
    def image_callback(self, msg):
        h, w = msg.shape[:2]
        self.image = msg
        tiles = []
        corners = []
        objects = []
        bridge = CvBridge()
        goal = darknetmsg.CheckForObjectsAction

        #self.index = int(self.count)%10
        if self.detect_ready:
            if self.index>2:
                im = msg[0:(h//4)*3,0:w]
                tile = cv2.resize(im,(w//3,h//3))
                self.index = 0
                i = 3
            else:
                i = self.index
                self.index +=1
                tile = msg[h//3:(h//3)*2,(w//3)*i:(w//3)*i+(w//3)]

            self.detect_ready = False
            #img = cv2.imread('/home/runar/boat_single.jpg')
            im = bridge.cv2_to_imgmsg(tile, 'bgr8')#encoding="passthrough")
            
            goal.id = i#self.count
            goal.image = im
            self.dark_client.send_goal(goal)
            #self.pub_tile.publish(im)
        else:
            self.show_webcam(msg)
            if self.count%10==0 and self.darknet is None:        # Start detection again if something fails
                self.detect_ready = True
        self.count += 1
        
        looking_angle = np.deg2rad(00+self.Mounting_angle*self.number)
        phi, theta, psi = self.euler_angles
 
        for d in self.detections:
            cv2.line(self.image, (d+int(w/2), 0), (d+int(w/2), h), (255,0,0), 10) 
        
        self.detections = []
        self.rotate_along_axis(-phi, -theta, -looking_angle)
        

    def show_webcam(self, image, corners=None):
        global initialize, boxToDraw#,tracker
        if boxToDraw is None:
            boxToDraw = corners
        elif corners is not None:# and boxToDraw is not None:
            iou = bb_intersection_over_union(boxToDraw,corners)
            if iou < 0.3:
                initialize = True
            if initialize:
                boxToDraw = corners
                initialize = False
                print("NEW_TRACK")
                boxToDraw = tracker.track(image[:,:,::-1], 'Cam', boxToDraw) 
            else:# all(corners)!=0:
                try:
                    boxToDraw = tracker.track(image[:,:,::-1], 'Cam')
                except:
                    print("No Bbox to track")
            if ((abs(boxToDraw[0]-boxToDraw[2]) > 4) and (abs(boxToDraw[1]-boxToDraw[3]) > 4)):
                        cv2.rectangle(self.image, (int(boxToDraw[0]), int(boxToDraw[1])), (int(boxToDraw[2]), int(boxToDraw[3])), 
                            [0,0,255], 2)
                        self.bb_angle = self.fov_pixel*(int(boxToDraw[0])+(int(boxToDraw[2])-int(boxToDraw[0]))/2-self.width/2)+self.number*np.deg2rad(self.Mounting_angle)
                        self.data_assosiation()
                        print('BoundingBox angle: ',self.bb_angle, np.rad2deg(self.bb_angle))
             
        
    def rotate_along_axis(self, phi=0, theta=0, psi=0, dx=0, dy=0, dz=0):
        # Get ideal focal length on z axis
        dz = self.focal*1.
        #axis = np.float32([[3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)

        # Get projection matrix
        mat = self.get_M(phi, theta, psi, dx, dy, dz)

        self.warp = cv2.warpPerspective(self.image, mat, (self.width, self.height))
        cv2.imshow('Cam3', self.warp)
        cv2.waitKey(2)
        

    """ Get Perspective Projection Matrix """
    def get_M(self, phi, theta, psi, dx=0, dy=0, dz=0):
        
        w = self.width
        h = self.height
        f = self.focal

        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, 1]])

        # Transform from image coordinate to body
        CB = np.array([ [0, 1, 0, 0],
                        [0, 0, -1, 0],
                        [-1, 0, 0, 0],
                        [0, 0, 0, 1]])
        
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
        R = np.dot(np.dot(RX, RY), RZ)
        #print(R,RX,RY,RZ)

        # Translation matrix
        T = np.array([  [1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([ [f, 0, w/2, 0],
                        [0, f, h/2, 0],
                        [0, 0, 1, 0]])

        # Composed rotation matrix with (CB, R, CBinv)
        RCB = np.dot(np.dot(CB,R),CB.transpose())

        # Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(RCB, A1)))
        #return R

    def draw(self, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = self.image.copy()
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        return img

    
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


    def Horizon(self):
        #lines = []
        #imgOg = cv2.imread(str(directory)+image) # Read image
        reduced = cv2.resize(self.image, (400, 400), interpolation = cv2.INTER_AREA)
        img_gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img_gray,(9,9),0)            # Perform a gaussian blur
        edges = cv2.Canny(blur,50,150,apertureSize = 3)
        #lines = cv2.HoughLinesP(edges,1,np.pi/180,100,80,10)
        lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 100,minLineLength = 100,maxLineGap = 50)
        
        try:
        #for i in range(N):
            for x1,y1,x2,y2 in lines[0]:
                cv2.line(reduced,(x1,y1),(x2,y2),(0,255,0),2)
                ang = np.arctan2((y2-y1),(x2-x1))
                phi, theta, psi = self.euler_angles
                with open('somefile.txt', 'a') as the_file:
                    #the_file.write('Hello\n')
                    the_file.write(str(self.pose_stamp)+';'+str(self.imagetimestamp)+';'+str(ang)+';'+str(phi)+'\n') 
        except:
            print('')
        cv2.imshow('Cam2',reduced)
        
          
    def start(self):
        #self.net = dn.load_net(b"/home/runar/yolov3.cfg", b"/home/runar/yolov3.weights", 0)
        #self.meta = dn.load_meta(b"/home/runar/coco.data")
        
        #rospy.loginfo("In attesa")

        self.number = 0
        self.Mounting_angle = 72       # 5 cameras, 360/5=72

        im_dir = "/home/runar/Skrivebord/0"
        file_list = os.listdir(im_dir)
        sorted_file_list = sorted(file_list)#, key=lambda x:x[-30:])
        i = 1#4300
        self.cam = cv2.VideoCapture('/home/runar/Ladybug/output0.mp4')
        #self.cam.set(1, 17000)
        
        self.fov_radians = np.deg2rad(100)      #FOV is about 100 deg
        self.fov_pixel = self.fov_radians/1616#self.width
        self.image_time = str(sorted_file_list[i])
        self.imageNametoTimestamp(self.image_time)
 
        while not rospy.is_shutdown():
            if (self.imagetimestamp - self.pose_stamp) < -2:
                i += int(abs(self.imagetimestamp - self.pose_stamp))
                self.image_time = str(sorted_file_list[i])
                #print("WAY SMALLER",self.imagetimestamp-self.pose_stamp, i) 
                self.imageNametoTimestamp(self.image_time)
            elif (self.imagetimestamp - self.pose_stamp) < -0.06:
                i += 2# + int(abs(self.imagetimestamp - self.pose_stamp))
                self.image_time = str(sorted_file_list[i])
                #print("SMALLER",self.imagetimestamp-self.pose_stamp, i) 
                self.imageNametoTimestamp(self.image_time)
            elif (self.imagetimestamp - self.pose_stamp) > 10:#0.06:
                i = 1       # Restart due to bag file restarted
                #print('LARGER',self.imagetimestamp-self.pose_stamp, i)
            else:
                i += 1# + int(abs(self.imagetimestamp - self.pose_stamp))
                self.image_time = str(sorted_file_list[i])
                #print("GOOD",self.imagetimestamp-self.pose_stamp, i) 
                self.imageNametoTimestamp(self.image_time)

            image = cv2.imread(im_dir + '/' + sorted_file_list[i])
            #ret_val, image = self.cam.read()

            if image is None:
                # End of video.
                print('No image')
            else:
                h, w = image.shape[:2]
                self.focal = 1350
                #mtx = np.matrix('1350.41716 0.0 1038.58110; 0.0 1352.74467 1219.10680; 0.0 0.0 1.0')
                self.mtx = np.matrix('1350.0 0.0 1024.0; 0.0 1350.0 1232.0; 0.0 0.0 1.0')
                #distort = np.array([-0.293594324, 0.0924910801, -0.000795067830, 0.000154218667, -0.0129375553])
                self.distort = np.array([-0.29, 0.09, -0.0, 0.0, -0.013])

                self.newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.mtx,self.distort,(w,h),1,(w,h))

                # crop the image
                cropx, cropy = [216, 600]

                dst = cv2.undistort(image, self.mtx, self.distort, None, self.newcameramtx)
                h, w = dst.shape[:2]
                image = dst[cropy:h-cropy, cropx:w-cropx]
                self.height, self.width = image.shape[:2]
 
                self.image_callback(image)

        cv2.destroyAllWindows()


# Main function
if __name__ == '__main__':
    #cv2.namedWindow('Cam', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('Cam2', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Cam3', cv2.WINDOW_NORMAL) 
    
    tracker = re3_tracker.Re3Tracker()
    rospy.init_node("CameraTracker")
    # Setup Telemetron ownship tracker
    #telemetron_tf = TransformListener()

    DetectObjects_node = DetectObjects()   
    DetectObjects_node.start()