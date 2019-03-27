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
#from cv_bridge import CvBridge, CvBridgeError
#import tf
#from tf import TransformListener
#import message_filters
from datetime import datetime
import matplotlib.pyplot as plt


basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from re3.tracker import re3_tracker 
from re3.tracker.darknet import darknet_orig as dn

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


boxToDraw = np.zeros(4)
initialize = True




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
    '''print ('DETECTTTTTTTTTTTTT',detect)
    cv2.imshow("Cam2",image)
    cv2.waitKey(200)
    '''
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
    '''
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
        self.count = 0
        self.q_b_w = []
        self.pos = []
        self.ned = []
        self.euler_angles = []
        self.tile = []
        self.tiles = []
        self.image = []
        self.radar_img = []
        self.millisecond = None
        self.firstimage = None
        self.second = None
        self.minute = None
        self.hour = None
        self.newimagetimestamp = 0
        self.radar_pixel = 0
        self.detections = []
        self.cdetections = []
        #self.dimg = []
        #self.bridge = CvBridge()
        #with np.load('calib.npz') as X:
        #    self.mtx, self.dist, self.rvec, self.tvec = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
        #print(self.mtx, self.dist, self.rvec, self.tvec)

        # Node cycle rate (in Hz).
        self.rate = rospy.Rate(10)

        # Publishers
        self.pub_tile = rospy.Publisher('/ladybug/object_img/image_calib', Image, queue_size=15)
        self.pub_calib = rospy.Publisher('/ladybug/calib_img/image_calib', Image, queue_size=5)

        # Subscribers
        rospy.Subscriber('/seapath/pose',geomsg.PoseStamped, self.pose_callback)
        #scan_topic = rospy.get_param('~scan_topic', 'radar_scans')
        #rospy.Subscriber('/radar/radar_scans', automsg.RadarScan, self.radar_callback)#, callback_args=(track_publisher, track_manager, telemetron_tf, measurement_covariance_parameters), queue_size=30)
        rospy.Subscriber('/radar/estimates', automsg.RadarEstimate, self.radar_callback)
        rospy.Subscriber('/radar/clusters', automsg.RadarCluster, self.cluster_callback)
        #rospy.Subscriber('/mr/spokes', automsg.RadarSpoke, self.spoke_callback )

        #self.pose = message_filters.Subscriber('/seapath/pose',geomsg.PoseStamped)
        #rospy.Subscriber('/ladybug/camera'+str(self.number)+'/image_raw', Image, self.image_callback)
        self.cam = cv2.VideoCapture('/home/runar/Ladybug/output0.mp4')
        self.cam.set(1, 17000-1)
        #ret_val, img = self.cam.read()
        # try:
        #     img = self.bridge.imgmsg_to_cv2(rosimg, "bgr8")
        #     img = self.bridge.imgmsg_to_cv2(rosimg, desired_encoding="passthrough")
        # except CvBridgeError as e:
        #     print(e)


            
            #ts = message_filters.ApproximateTimeSynchronizer([self.pose, self.image], 10, 10, allow_headerless=True)
            #ts = message_filters.TimeSynchronizer([self.pose, self.image], 10)
            #ts.registerCallback(self.callback)

    def pose_callback(self, msg):
        self.pose_stamp = float(str(msg.header.stamp))/1e9
        self.pose_time = datetime.fromtimestamp(self.pose_stamp)
        self.position = msg.pose.position
        self.quat = msg.pose.orientation
        self.ned = np.array([self.position.x, self.position.y, self.position.z])
        self.q_b_w = np.array([self.quat.x, self.quat.y, self.quat.z, self.quat.w])
        self.euler_angles = conv.quaternion_to_euler_angles(self.q_b_w)
        #print('ANGLES', self.euler_angles)
        #print(self.mtx, self.dist, self.rvec, self.tvec)

    def radar_callback(self, msg):
        
        self.radar_stamp = float(str(msg.header.stamp))/1e9
        self.est_track_id = msg.track_id
        self.posterior_pos = msg.posterior.pos_est
        self.posterior_vel = msg.posterior.vel_est
        #print(self.posterior_pos, self.posterior_vel)
        self.posterior_ned = np.array([self.posterior_pos.x, self.posterior_pos.y])
        #print(self.radar_stamp)
        #print(self.centroid_ned, self.ned)
        dx = self.posterior_pos.x - self.position.x
        dy = self.posterior_pos.y - self.position.y
        phi, theta, psi = self.euler_angles

        radar_angle_ned = np.arctan2(dx,dy)
        self.radar_angle_body = radar_angle_ned - psi + np.deg2rad(3)     # Installation angle offset between camera and radar  
        if self.radar_angle_body < -np.pi:
            self.radar_angle_body += 2*np.pi
        elif self.radar_angle_body > np.pi:
            self.radar_angle_body -= 2*np.pi
        self.radar_pixel = int(self.radar_angle_body/self.fov_pixel)
        #if abs(self.radar_pixel) < (self.width/2):
        #self.detections.append(self.radar_pixel)
            #self.radar_img = self.warp.copy()
            #cv2.line(self.warp, (self.radar_pixel, 0), (self.radar_pixel, self.height), (0,255,0), 10)
            #cv2.waitKey(100)
        #print('RADAR_ANGLE',radar_angle_ned, psi, self.radar_angle_body, self.radar_pixel, self.width)
        self.radar_range = np.sqrt(dx**2+dy**2)
        print(self.radar_range, self.est_track_id, np.rad2deg(self.radar_angle_body), self.radar_pixel)

        #print(msg)
        #cv2.imshow('Cam3', self.warp)

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
        tiles = []
        corners = []
        objects = []

        cv2.imshow('Cam', msg)
        cv2.waitKey(1)

        ## Detect larger objects covering most of the image
        #tiles.append(msg[h//4:(h//4)*3,0:w])

        ## Detect smaller objects
        tiles.append(msg[h//3:(h//3)*2,0:(w//3)])
        tiles.append(msg[h//3:(h//3)*2,(w//3):(w//3)*2])
        tiles.append(msg[h//3:(h//3)*2,(w//3)*2:w])
        
        i = self.count%10
        if i < 3:
            tile = msg[h//3:(h//3)*2,(w//3)*i:(w//3)*i+(w//3)]
            #for tile in tiles:
                #cv2.imshow('Tile', tile)
                #cv2.waitKey(2000)
           
            p1p=detector(tile, self.net, self.meta)
            for d in p1p:
                print (d)
                xc, yc, w1, h1 = d[2]
                lst = list(d)
                #print(lst)
                lst[2] = ((xc+((w//3)*i)),(yc+((h//3))),w1,h1)
                #print(lst)
                objects.append(lst)

            corners.append(findobject(objects))
            #print("CORNERS",corners)
            if corners != [[]]:
                corners.sort(key=lambda x:x[1])
                corner = corners[0][2]
                self.show_webcam(self.image, corner)
            else:
                self.show_webcam(self.image)
                #corner = np.zeros(4)
        else:
            self.show_webcam(self.image)
        
        self.count += 1
        # Transform coordinate system
        Mounting_angle = 72       # 5 cameras, 360/5=72

        looking_angle = np.deg2rad(00+Mounting_angle*self.number)
        phi, theta, psi = self.euler_angles
        #print(self.euler_angles, looking_angle)
        #psi = looking_angle
        for d in self.cdetections:
            #print("DDDD  ",d)
            cv2.line(self.image, (d+int(w/2), 0), (d+int(w/2), h), (0,255,0), 10) 
        for d in self.detections:
            #print("CCCC  ",d)
            cv2.line(self.image, (d+int(w/2), 0), (d+int(w/2), h), (255,0,0), 10) 
        self.detections = []
        self.cdetections = [] 
        self.rotate_along_axis(-phi, -theta, -looking_angle)
        #print(corner)
    
    def show_webcam(self, image, corners=None):
        global initialize, boxToDraw#,tracker

        if corners is None:
            try:
                boxToDraw = self.tracker.track(image[:,:,::-1], 'Cam')
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
                boxToDraw = self.tracker.track(image[:,:,::-1], 'Cam', boxToDraw)

        elif ((abs(boxToDraw[0]-boxToDraw[2]) > 5) and (abs(boxToDraw[1]-boxToDraw[3]) > 5)):
            boxToDraw = self.tracker.track(image[:,:,::-1], 'Cam')
            cv2.rectangle(self.image,
                (int(boxToDraw[0]), int(boxToDraw[1])),
                (int(boxToDraw[2]), int(boxToDraw[3])),
                [0,0,255], 2)

        #else:
        #    print("TERMINATED")
        
        cv2.imshow('Cam', self.image)
        #print (boxToDraw)        


    """ Wrapper of Rotating a Image """
    def rotate_along_axis(self, phi=0, theta=0, psi=0, dx=0, dy=0, dz=0):
        
        # Get radius of rotation along 3 axes
        #rtheta, rphi, rpsi = np.deg2rad(theta, phi, psi)
        
        # Get ideal focal length on z axis
        # NOTE: Change this section to other axis if needed
        #d = np.sqrt(self.height**2 + self.width**2)
        #self.focal = d / (2 * np.sin(psi) if np.sin(psi) != 0 else 1)
        dz = self.focal*1.
        axis = np.float32([[3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)

        # Get projection matrix
        mat = self.get_M(phi, theta, psi, dx, dy, dz)
        #print (mat)
        '''
        #gray = cv2.cvtColor(self.image.copy(),cv2.COLOR_BGR2GRAY)
        corners = np.array([[1000,1000],[1000,2000],[2000,1000],[2000,2000]])
        #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        #corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        rvec, jacobian = cv2.Rodrigues(mat)
        tvec = np.array([0.,0.,1.])
        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, self.mtx, self.distort)
        imgpts = imgpts.reshape(-1,2)
        im = self.draw(corners,imgpts)
        #print(imgpts,corners)
        #p_start = (10, self.height/2)
        #p_stop = (self.width-10, self.height/2)
        #cv2.line(self.image, p_start, p_stop, (0,0,255), 3) #Red
        cv2.imshow('Cam2', im)
        cv2.waitKey(1)
        '''
        self.warp = cv2.warpPerspective(self.image, mat, (self.width, self.height))
        cv2.imshow('Cam3', self.warp)
        cv2.waitKey(1)
        #return self.warp

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

        '''# Transform from image coordinate to body
        CB = np.array([ [0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])
        '''
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
            #print(self.firstimagetimestamp)
            #self.imagetimestamp == self.firstimagetimestamp
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
            #self.firstimagetimestamp = self.newimagetimestamp
            print('Timestamp updated')
        #print((milli-self.firstimage)/10., self.firstimagetimestamp)
        '''
        if self.hour == None or self.hour < hr:
            self.hour = hr
        if self.minute == None or self.minute < mi:
            self.minute = mi
         
        if (self.second) == None:# or self.second < sec:
            self.second = sec
        
        if (self.millisecond) == None:
            self.millisecond = milli
        if self.millisecond != milli: 
            self.second += 0.1
        self.second = 
        #print('MILLI',self.millisecond)
        if (self.second) > 59:
            self.second = 0
            self.minute +=1
        if self.minute > 59:
            self.minute = 0
            self.hour +=1
        if self.hour > 23:
            self.hour = 0
        
        #print(self.second%1)
        
        imagetime = datetime(self.year, self.month, self.day, self.hour, self.minute, int(self.second), int((self.second%1)*1000))
        #print('IMAGETIME',imagetime)
        self.imagetimestamp = time.mktime(imagetime.timetuple())
        
        
        #return self.imagetimestamp 
        '''
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
        self.net = dn.load_net(b"/home/runar/yolov3.cfg", b"/home/runar/yolov3.weights", 0)
        self.meta = dn.load_meta(b"/home/runar/coco.data")
        self.tracker = re3_tracker.Re3Tracker()
        #rospy.loginfo("In attesa")
        #print("FOV   ",self.fov_pixel)
        self.number = 0
        im_dir = "/home/runar/Skrivebord/0"
        file_list = os.listdir(im_dir)
        sorted_file_list = sorted(file_list)#, key=lambda x:x[-30:])
        i = 1#4300
        #self.image = cv2.imread(im_dir + '/' + sorted_file_list[i])
        #self.height, self.width = self.image.shape[:2]
        #print('WWWWWWW', self.width)
        self.fov_radians = np.deg2rad(100)      #FOV is about 100 deg
        self.fov_pixel = self.fov_radians/2048#1616.#self.width
        self.image_time = str(sorted_file_list[i])
        #print(self.image_time) 
        self.imageNametoTimestamp(self.image_time)
        #print(self.imagetimestamp-self.pose_stamp, i) 
        while not rospy.is_shutdown():
            #ret_val, self.image = self.cam.read()
            
            #for i in range(len(sorted_file_list)):# glob.glob('/home/runar/Skrivebord/3/*jpg'):
            #self.image = cv2.imread(im_dir + '/' + sorted_file_list[i])
            
            #print('Ladybug', self.imagetimestamp)
            #print('POSE   ', self.pose_stamp)
            #if abs(self.imagetimestamp - self.pose_stamp) > 0.:
            '''
            if (self.imagetimestamp - self.pose_stamp) < -500:
                i += 500# + int(abs(self.imagetimestamp - self.pose_stamp))
                self.image_time = str(sorted_file_list[i])
                print("WAIT FOR ME, SMALLER",self.imagetimestamp-self.pose_stamp, i) 
                self.imageNametoTimestamp(self.image_time)
            elif (self.imagetimestamp - self.pose_stamp) < -50:
                i += 50# + int(abs(self.imagetimestamp - self.pose_stamp))
                self.image_time = str(sorted_file_list[i])
                print("WAY WAY SMALLER",self.imagetimestamp-self.pose_stamp, i) 
                self.imageNametoTimestamp(self.image_time)
            '''
            if (self.imagetimestamp - self.pose_stamp) < -2:
                i += int(abs(self.imagetimestamp - self.pose_stamp))
                self.image_time = str(sorted_file_list[i])
                print("WAY SMALLER",self.imagetimestamp-self.pose_stamp, i) 
                self.imageNametoTimestamp(self.image_time)
            elif (self.imagetimestamp - self.pose_stamp) < -0.06:
                i += 2# + int(abs(self.imagetimestamp - self.pose_stamp))
                self.image_time = str(sorted_file_list[i])
                print("SMALLER",self.imagetimestamp-self.pose_stamp, i) 
                self.imageNametoTimestamp(self.image_time)
            elif (self.imagetimestamp - self.pose_stamp) > 0.06:
                print('LARGER',self.imagetimestamp-self.pose_stamp, i)
            else:
                i += 1# + int(abs(self.imagetimestamp - self.pose_stamp))
                self.image_time = str(sorted_file_list[i])
                print("GOOD",self.imagetimestamp-self.pose_stamp, i) 
                self.imageNametoTimestamp(self.image_time)
                #while self.imagetimestamp > self.pose_stamp:
                #    break
            self.image = cv2.imread(im_dir + '/' + sorted_file_list[i])
            
            if self.image is None:
                # End of video.
                print('No image')
            
            else:
                #cv2.imshow('Cam', self.image)
                #cv2.waitKey(1)
                h, w = self.image.shape[:2]
                #print(h)

                self.focal = 1350
                #mtx = np.matrix('1350.41716 0.0 1038.58110; 0.0 1352.74467 1219.10680; 0.0 0.0 1.0')
                self.mtx = np.matrix('1350.0 0.0 1024.0; 0.0 1350.0 1232.0; 0.0 0.0 1.0')
                #distort = np.array([-0.293594324, 0.0924910801, -0.000795067830, 0.000154218667, -0.0129375553])
                self.distort = np.array([-0.29, 0.09, -0.0, 0.0, -0.013])

                self.newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.mtx,self.distort,(w,h),1,(w,h))

                # crop the image
                self.cropx, self.cropy = [216, 600]

                self.dst = cv2.undistort(self.image, self.mtx, self.distort, None, self.newcameramtx)
                h, w = self.dst.shape[:2]
                self.image = self.dst[self.cropy:h-self.cropy, self.cropx:w-self.cropx]
                self.height, self.width = self.image.shape[:2]
                #self.image = self.bridge.cv2_to_imgmsg(self.cv_image, encoding="passthrough")
                #self.image.header.stamp = rospy.Time.now()

                #cv2.imshow('Cam', self.image)
                #self.Horizon()
                #print("IMGSIZE",ret_val,cropx,cropy,w,h, roi, newcameramtx)
                #h, w = self.dimg.shape[:2]
                #im = dimg.copy()  
                self.image_callback(self.image)

            #i += 1
            keyPressed = cv2.waitKey(1)
            if keyPressed == 27 or keyPressed == 1048603:
                break  # esc to quit
            elif keyPressed != -1:
                paused = True
            #frameNum += 1
            self.rate.sleep()
            '''
            self.pub_calib.publish(self.image)
            for self.tile in self.tiles:
            #print(self.q_b_w, self.euler_angles,self.ned)
                self.pub_tile.publish(self.tile)
            #self.loop_rate.sleep()
            self.tiles = []
            self.tile = []
            '''
            #print("running")


# Main function
if __name__ == '__main__':
    cv2.namedWindow('Cam', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('Cam2', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Cam3', cv2.WINDOW_NORMAL) 
    
    rospy.init_node("CameraTracker")
    # Setup Telemetron ownship tracker
    #telemetron_tf = TransformListener()

    DetectObjects_node = DetectObjects()   
    DetectObjects_node.start()

    #CorrectImage_node = CorrectImage()
    #CorrectImage_node.start()
    '''
    #rospy.Subscriber('/seapath/pose',geomsg.PoseStamped, publish_seapath_pose)
    cam1 = rospy.Subscriber('/ladybug/camera0/image_raw', Image, queue_size=1)
    cam2 = rospy.Subscriber('/ladybug/camera1/image_raw', Image, queue_size=1)
    cam2 = rospy.Subscriber('/ladybug/camera2/image_raw', Image, queue_size=1)
    cam2 = rospy.Subscriber('/ladybug/camera3/image_raw', Image, queue_size=1)
    cam2 = rospy.Subscriber('/ladybug/camera4/image_raw', Image, queue_size=1)
    campub = rospy.Publisher('/ladybug/object_img/image_raw', Image, queue_size=15)
    selfAttitude = rospy.Subscriber('/seapath/euler_angles', stdmsg.Float64, queue_size=2)
    selfPosition = rospy.Subscriber('/seapath/position', stdmsg.Float64, queue_size=2)
    #ros_parameters = rospy.get_param('~')
    #ladybug_parameters = ros_parameters['ladybug']
    #measurement_covariance_parameters = ros_parameters['measurement_covariance']
    
    #cam = cv2.VideoCapture('/home/runar/re3-tensorflow/demo/Bar.mp4')
    cam = cv2.VideoCapture('/home/runar/Ladybug/output0.mp4')
    cam.set(1, 6200-1)
    cv2.namedWindow('Cam', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Cam2', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('Cam3', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('Cam4', cv2.WINDOW_NORMAL)
    ret_val, img = cam.read()
    h, w = img.shape[:2]

    #mtx = np.matrix('1350.41716 0.0 1038.58110; 0.0 1352.74467 1219.10680; 0.0 0.0 1.0')
    mtx = np.matrix('1350.0 0.0 1024.0; 0.0 1350.0 1232.0; 0.0 0.0 1.0')
    #distort = np.array([-0.293594324, 0.0924910801, -0.000795067830, 0.000154218667, -0.0129375553])
    distort = np.array([-0.29, 0.09, -0.0, 0.0, -0.013])

    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,distort,(w,h),1,(w,h))

    # crop the image
    cropx, cropy = [216, 600]
    #img = dst[y:y+h, x:x+w]
    
    #tracker = re3_tracker.Re3Tracker()
    #server = DetectObjects(rospy.get_name())

    #detector = "./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights /media/runar/Seagate\ Expansion\ Drive/Ladybug/1/output.mp4 -i 0"
    #net = dn.load_net(b"/home/runar/yolov3.cfg", b"/home/runar/yolov3.weights", 0)
    #meta = dn.load_meta(b"/home/runar/coco.data")
    #img = cv2.imread('/home/runar/boats.jpg')
    frameNum = 0
    print(frameNum)
    #scan_topic = rospy.get_param('~scan_topic', 'radar_scans')
    #rospy.Subscriber(scan_topic, automsg.RadarScan, callback=scan_callback, callback_args=(track_publisher, track_manager, telemetron_tf, measurement_covariance_parameters), queue_size=30)
    #print(euler_angles, pos)
    #rospy.spin()
    '''
    '''
    while True:
        print(frameNum)
    
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
                p1p=detector(tile, i)#, net, meta)
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

    #while not rospy.is_shutdown():
    #    rospy.spin()
    '''    
    cv2.destroyAllWindows()