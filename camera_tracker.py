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
from munkres import Munkres, print_matrix, make_cost_matrix
import numdifftools as nd


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


np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)


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

class ExtendedKalman():
    """
    Nonlinear Kalman Filter Implementation
    """
    def __init__(self,initialObservation, numSensors, sensorNoise, stateTransitionFunction, sensorTransferFunction, processNoise, processNoiseCovariance):
        self.numSensors = numSensors
        self.estimate = initialObservation  #current estimate, initialized with first observation
        self.previousEstimate = initialObservation  #previous state's estimate, initialized with first observation
        self.gain = np.identity(numSensors) #tradeoff of system between estimation and observation, initialized arbitrarily at identity
        self.previousGain = np.identity(numSensors)  #previous gain, again arbitarily initialized
        self.errorPrediction = np.identity(numSensors) #current estimation of the signal error,... starts as identity arbitrarily
        self.previousErrorPrediction = np.identity(numSensors)  #previous signal error, again arbitrary initialization
        self.sensorNoiseProperty = sensorNoise #variance of sensor noise
        self.f = stateTransitionFunction #state-transition function, from user input
        self.fJac = nd.Jacobian(self.f) #jacobian of f
        self.h = sensorTransferFunction  #sensor transfer function, from user input
        self.hJac = nd.Jacobian(self.h) #jacobian of h
        self.processNoise = processNoise;  #process noise
        self.Q = processNoiseCovariance #process noise covariance

    def predict(self):
        """
        Called first.
        Predicts estimate and error prediction according to model of the situation
        """
        #update current state
        self.estimate = self.f(self.previousEstimate) + self.processNoise

        #find current jacobian value
        jacVal = self.fJac(self.previousEstimate) + self.processNoise

        #update error prediction state
        self.errorPrediction = np.dot(jacVal , np.dot(self.previousErrorPrediction,np.transpose(jacVal))) + self.Q

    def update(self,currentObservation):
        """
        Called second.
        Updates estimate according to combination of observed and prediction.
        Also updates our learning parameters of gain and errorprediction.
        """
        #update the current estimate based on the gain
        self.estimate = self.estimate + np.dot(self.gain,(currentObservation - self.h(self.estimate)))
        #find current jacobian value
        jacVal = self.hJac(self.estimate)

        #update the gain based on results from hte previous attempt at estimating
        invVal = np.dot(jacVal, np.dot(self.errorPrediction, np.transpose(jacVal))) + self.sensorNoiseProperty
        self.gain = np.dot(self.errorPrediction, np.dot(np.transpose(jacVal) , np.linalg.inv(invVal) ))
        #update error prediction based on our success
        self.errorPrediction = np.dot((np.identity(self.numSensors) - np.dot(self.gain, jacVal)), self.errorPrediction)

        #update variables for next round
        self.previousEstimate = self.estimate
        self.previousGain = self.gain
        self.previousErrorPrediction = self.errorPrediction;

    def getEstimate(self):
        """
        Simple getter for cleanliness
        """
        return self.estimate


    '''
    #variables governing the simulation
    numSamples = 100

    #our sensor simulators ... voltmeter and ammeter
    voltmeter = sensors.Voltmeter(0,3)
    ammeter = sensors.Ammeter(0,2)
    #and their associated state transfer functions, sensor transfer functions, and noise values
    stateTransfer = lambda x: np.array([[math.pow(x[0][0],1.01)],[math.pow(x[1][0],.99)+5]]) 
    sensorTransfer = lambda x: x 
    sensorNoise = np.array([[math.pow(3,2),0],[0,math.pow(2,2)]])
    processNoise = np.array([[0],[0]])
    processNoiseCovariance = np.array([[.1,0],[0,.1]])

    #result log holders
    x_vals = []
    volt_vals = []
    current_vals = []
    r_vals = []
    ekfv_vals = []
    ekfc_vals = []
    ekfr_vals = []

    #finally grab initial readings
    voltVal = voltmeter.getData()
    currentVal = ammeter.getData()
    #put them in a column vector
    initialReading = np.array([[voltVal],[currentVal]])  #values are column vectors
    #and initialize our filter with our initial reading, our 2 sensors, and all of the associated data
    kf = ExtendedKalman(initialReading,2,sensorNoise,stateTransfer,sensorTransfer, processNoise, processNoiseCovariance)

    #now run the simulation
    for i in range(numSamples)[1:]:
        #grab data
        voltVal = voltmeter.getData() 
        currentVal = ammeter.getData()  
        reading = np.array([[voltVal],[currentVal]])  #values are column vectors

        #predict & update
        kf.predict()
        kf.update(reading)

        #grab result for this iteration and figure out a resistance value
        myEstimate = kf.getEstimate()
        voltage_guess = myEstimate[0][0]
        current_guess = myEstimate[1][0]
        current_resistance = voltage_guess / current_guess



    '''
class DetectObjects(object):
    def __init__(self):
        # Params
        #self.darknet = None
        self.detect_ready = True
        self.count = 0
        self.index = 0
        #self.q_b_w = []
        self.pos = []
        #self.ned = []
        self.euler_angles = [0,0,0]
        #self.tile = []
        #self.tiles = []
        self.newimage = False
        self.image = []
        self.warp = []
        self.image3 = []
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
        self.range = 100
        self.track_id = None
        self.bb_angle = 0

        self.focal = 1350
        self.number = 4
        self.Mounting_angle = 72       # 5 cameras, 360/5=72
        self.fov_radians = np.deg2rad(100)      #FOV is about 100 deg
        self.fov_pixel = self.fov_radians/1616#self.width

        self.matrix = None
        self.penalty = np.zeros([100])

        # Estimation parameter of EKF
        self.Q = np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0])**2  # predict state covariance
        self.R = np.diag([1.0, 1.0])**2  # Observation x,y position covariance

        #self.dimg = []
        #self.bridge = CvBridge()
        #with np.load('calib.npz') as X:
        #    self.mtx, self.dist, self.rvec, self.tvec = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
        #print(self.mtx, self.dist, self.rvec, self.tvec)

        # Node cycle rate (in Hz).
        #self.rate = rospy.Rate(10)

        # Publishers
        #self.pub_tile = rospy.Publisher('/ladybug/object_img/image_raw', Image, queue_size=1)
        #self.pub_goal = rospy.Publisher('/darknet_ros/check_for_objects/goal', darknetmsg.CheckForObjectsActionGoal, queue_size=1)
        #self.dark_client = actionlib.SimpleActionClient('darknet_ros/check_for_objects', darknetmsg.CheckForObjectsAction)
        #self.pub_calib = rospy.Publisher('/ladybug/calib_img/image_calib', Image, queue_size=5)

        # Subscribers
        #rospy.Subscriber('/seapath/pose',geomsg.PoseStamped, self.pose_callback)
        #scan_topic = rospy.get_param('~scan_topic', 'radar_scans')
        #rospy.Subscriber('/radar/radar_scans', automsg.RadarScan, self.radar_callback)#, callback_args=(track_publisher, track_manager, telemetron_tf, measurement_covariance_parameters), queue_size=30)
        #rospy.Subscriber('/radar/estimates', automsg.RadarEstimate, self.radar_callback)
        #rospy.Subscriber('/radar/clusters', automsg.RadarCluster, self.cluster_callback)
        #rospy.Subscriber('/mr/spokes', automsg.RadarSpoke, self.spoke_callback )
        #rospy.Subscriber('/darknet_ros/found_object', stdmsg.Int8, self.darknet2_callback)
        #rospy.Subscriber('/darknet_ros/bounding_boxes',darknetmsg.BoundingBoxes, self.darknet_callback)
        #rospy.Subscriber('/darknet_ros/check_for_objects/result',darknetmsg.CheckForObjectsActionResult, self.darknet3_callback)
        #rospy.Subscriber('/darknet_ros/detection_image', Image, self.darknet_detect_image)

        #self.pose = message_filters.Subscriber('/seapath/pose',geomsg.PoseStamped)
        #rospy.Subscriber('/ladybug/camera0/image_raw', Image, self.image_callback)
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

        EKF_node = ExtendedKalman()   
        #EKF_node.start()
        kf = EKF_node(initialReading,2,sensorNoise,stateTransfer,sensorTransfer, processNoise, processNoiseCovariance)

        #predict & update
        kf.predict()
        kf.update(reading)

        #grab result for this iteration and figure out a resistance value
        myEstimate = kf.getEstimate()

        '''
        # State Vector [x y yaw v]'
        self.xEst = np.zeros((4, 1))
        self.xTrue = np.zeros((4, 1))
        self.PEst = np.eye(4)

        xDR = np.zeros((4, 1))  # Dead reckoning

        u = calc_input()
        xTrue, z, xDR, ud = self.observation(xTrue, xDR, u)
        xEst, PEst = self.ekf_estimation(xEst, PEst, z, ud)
        '''


    def data_assosiation(self):
        #print('DATA')
        a = []
        c = []
        e = []
        #try:
        if self.radar_detections != []:
            for d in self.radar_detections:
                #print('A')
                a.append([abs(int(d[0])-self.bb_angle), abs(int(d[1])-self.range), d[2], d])
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
                if b[0] < np.deg2rad(10):  # Only look in +/-10 degree area
                    c.append(b)
                    #print('B')
            try:
                c.sort(key=lambda x:x[1])
            except:
                print('c sort failed')
            for d in c:
                if d[2] == self.track_id:
                    e = [d]+e
                    #print('e', e)
                else:
                    e.append(d)
            try:
                self.range = e[0][1]
                self.track_id = e[0][2]
                print('RANGE: ',self.range, self.track_id)
                self.extended_kalman(e[0][3])
            except:
                print('data assosiation failed')
            
        #else:
         #   print('No radar detections')
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
        ident = str(msg.result.id)
        _,ident2,_ = (str(msg.status.goal_id.id)).split('-')
        status = msg.status.status


        corners = None
        corner = []
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
                for c in corners:
                    corner.append(np.array(c[0][2]))
                #print('Detected boat: ',corner)
                self.show_webcam(corner)
            else:
                self.show_webcam()

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
        print('DARKNET feedback:  ', msg)
        
        

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
        #print('RADAR',msg)
        self.radar_stamp = float(str(msg.header.stamp))/1e9
        #print(self.radar_stamp, self.pose_stamp)
        self.radar_track_id = msg.track_id
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
        self.radar_detections.append(np.array([self.radar_angle_body, self.radar_range, self.radar_track_id,
            [posterior_vel.x, posterior_vel.y], [posterior_pos_cov.var_x, posterior_pos_cov.var_y, posterior_pos_cov.cor_xy], 
            [posterior_vel_cov.var_x, posterior_vel_cov.var_y, posterior_vel_cov.cor_xy]]))
        self.detections.append(self.radar_pixel)

        #cv2.line(self.warp, (self.radar_pixel, 0), (self.radar_pixel, self.height), (0,255,0), 10)
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

    def call_server(self,msg):
        self.dark_client = actionlib.SimpleActionClient('darknet_ros/check_for_objects', darknetmsg.CheckForObjectsAction)
        self.dark_client.wait_for_server()
        bridge = CvBridge()
        im = bridge.cv2_to_imgmsg(msg, 'bgr8')#encoding="passthrough")
        goal = CheckForObjectsGoal()
        goal.id = self.index
        goal.image = im
        self.dark_client.send_goal(goal)#, feedback_cb=self.darknet_callback)
        self.dark_client.wait_for_result()
        result = self.dark_client.get_result()

        corners = None
        #corner = []
        h = self.height
        w = self.width
        self.dark_stamp = float(str(result.bounding_boxes.header.stamp))/1e9
        i = int(result.id)
        dark_boxes = result.bounding_boxes.bounding_boxes
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
                #print('Prob: ', prob)

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
        #print(corners)
        return corners


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
        
        

    def show_webcam(self, image, corners=None):
        box = {}
        #i = 0
        h = self.height
        w = self.width
        global initialize, boxToDraw#,tracker

        
        '''    
        if self.matrix is None:
            #boxToDraw = {}
            self.penalty = np.zeros([10])
            self.matrix = np.ones([10, 10])*1000#np.inf  ## max 10 different boats
            #for i in range(len(self.penalty)):
            #    boxToDraw[i] = [0,0,0,0]
        '''

        if corners is not None:
            if boxToDraw is None:# and all(corners) > 0:
                boxToDraw = {}
                i = 0
                for c in corners:
                    boxToDraw[i] = c
                    i += 1
            size = max(len(boxToDraw),len(corners))
            if size > len(self.penalty):
                size = len(self.penalty)
            self.matrix = np.ones([size, size])#np.inf  ## max 10 different boats
            print(len(boxToDraw),len(corners))
            for ckey in range(len(corners)):
                cvalue = corners[ckey]
                #print(cvalue)
                ckey_given_value = False
                new = False
                for bkey in boxToDraw:
                    if ckey_given_value:
                        break
                    self.b = bkey
                    bvalue = boxToDraw[bkey]
                    iou = 1
                    #print(bvalue)
                    #try:
                    #    if all(bvalue) == -1:
                    #        self.matrix[bkey][ckey] = 99
                    #        break
                    #except:
                    #    print()
                    if ((abs(bvalue[0]-bvalue[2]) > 4) and (abs(bvalue[1]-bvalue[3]) > 4)):
                        iou = bb_intersection_over_union(bvalue,cvalue)
                        if iou == 0:
                            #print(iou)
                            try:
                                new = True
                                self.penalty[bkey] += 1
                                #print('PENALTY',bkey,self.penalty)
                                if self.penalty[bkey] > 50:        # If no detections for X iterations. Remove track
                                    boxToDraw[bkey] = [0,0,0,0]
                                    box[bkey] = [0,0,0,0]
                                    self.penalty[bkey] = 0
                                self.matrix[bkey][ckey] = 0
                            except:
                                break#print('OverFlow')
                        else:
                            new = False
                            try:
                                self.matrix[bkey][ckey] = -iou
                                self.penalty[bkey] = 0
                            except:
                                break#print('OVERflow')
                    else:
                        try:
                            boxToDraw[bkey] = corners[ckey]
                            self.matrix[bkey][ckey] = -1
                            ckey_given_value = True
                            new = False
                        except:
                            break#print('overFLOW')
                            
                if new:
                    boxToDraw[self.b+1] = corners[ckey]
                    box[self.b+1] = corners[ckey]
                    if self.matrix.shape[0] < len(boxToDraw):
                        self.matrix = self.matrix[..., np.newaxis]
                    try:
                        self.matrix[self.b+1][ckey] = -0.5
                    except:
                        break#print('overflow')


            m = Munkres()
            #print(self.matrix)
            indexes = m.compute(self.matrix)
            #print(self.matrix)
            print(indexes)
            #size = min(len(boxToDraw),len(corners))
            for a,b in indexes:
                if a > len(corners)-1:
                    break
                #elif b > len(boxToDraw)-1 and boxToDraw[b] != [0,0,0,0]:
                #    box[b] = corners[a]
                else:
                    boxToDraw[b] = corners[a]

                if self.matrix[a][b] < -0.2:
                    self.penalty[b] = 0
                elif self.matrix[a][b] > 0:
                    print('no_good')
                else:
                    box[b] = corners[a]
                    for c in boxToDraw:
                        if c != b:
                            iou = bb_intersection_over_union(box[b],boxToDraw[c])
                            if iou > 0.3:
                                boxToDraw[c] = [0,0,0,0]
                                box[c] = [0,0,0,0]

                #except:
                #    print()

            bboxes = tracker.multi_track(image[:,:,::-1], boxToDraw.keys(), box)
            #print('From tracker',bboxes)
            i = 0
            for b in boxToDraw:
                boxToDraw[b] = bboxes[i]
                i += 1

        else:
            try:
                box = boxToDraw
                for bkey in boxToDraw:
                    self.penalty[bkey] += 1
                    if self.penalty[bkey] > 50:
                        boxToDraw[bkey] = [0,0,0,0]
                        del box[bkey]
                        self.penalty[bkey] = 0
                bboxes = tracker.multi_track(image[:,:,::-1], box.keys())
                i = 0
                for b in box:
                    boxToDraw[b] = bboxes[i]
                    i += 1
            except:
                print("No Bbox to track")
        if boxToDraw is not None:
            if any(boxToDraw) != 0:
                for a in boxToDraw:
                    b = boxToDraw[a]
                    #print(a,b)
                    if ((abs(b[0]-b[2]) > 4) and (abs(b[1]-b[3]) > 4)):
                        cv2.rectangle(self.draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), 
                            [0,0,255], 2)
                        self.bb_angle = self.fov_pixel*(int(b[0])+(int(b[2])-int(b[0]))/2-self.width/2)+self.number*np.deg2rad(self.Mounting_angle)
                        #self.data_assosiation()

                    


            
    def pixel_2_angle(self, x, y, z=1):
        x_angle = x*self.fov_pixel-(self.width/2)*self.fov_pixel
        y_angle = y*self.fov_pixel-(self.height/2)*self.fov_pixel
        z_angle = z
        return [x_angle, y_angle, x_angle] 

    def angle_2_pixel(self, rx, ry, rz=0):
        x = (rx/self.fov_pixel)+(self.width/2)-np.deg2rad(self.Mounting_angle*self.fov_pixel)
        y = (ry/self.fov_pixel)+(self.height/2)
        z = 1
        return [x, y, z]
        
    def rotate_along_axis(self, image, phi=0, theta=0, psi=0, dx=0, dy=0, dz=0):
        self.warppose = self.euler_angles
        # Get ideal focal length on z axis
        dz = self.focal*1.
        #axis = np.float32([[3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)

        # Get projection matrix
        mat = self.get_M(phi, theta, psi, dx, dy, dz)
        self.warp = image#cv2.warpPerspective(image, mat, (self.width, self.height))
        self.draw = self.warp.copy()
        #print('WARP')
        

    """ Get Perspective Projection Matrix """
    def get_M(self, phi, theta, psi, dx=0, dy=0, dz=None, w=None, h=None, f=None):
        d1 = 0
        #CB = np.eye(4)
        if w is None:
            w=self.width
            h=self.height
            f=self.focal
        if dz is None:
            dz = self.focal
            d1 = 1
        
        # Transform from image coordinate to body
        CB = np.array([ [0, 1, 0, 0],
                        [0, 0, -1, 0],
                        [-1, 0, 0, 0],
                        [0, 0, 0, 1]])

        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, d1]])
        
        R = self.rotation_3D(phi, theta, psi)

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
        #print(R)
        # Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(RCB, A1)))
        #return R

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
        # Subscribers
        rospy.Subscriber('/seapath/pose',geomsg.PoseStamped, self.pose_callback)
        rospy.Subscriber('/radar/estimates', automsg.RadarEstimate, self.radar_callback)
        rospy.Subscriber('/ladybug/camera0/image_raw', Image, self.image_callback)
        #self.net = dn.load_net(b"/home/runar/yolov3.cfg", b"/home/runar/yolov3.weights", 0)
        #self.meta = dn.load_meta(b"/home/runar/coco.data")
        
        #rospy.loginfo("In attesa")

        
        '''
        im_dir = "/home/runar/Skrivebord/0"
        file_list = os.listdir(im_dir)
        sorted_file_list = sorted(file_list)#, key=lambda x:x[-30:])
        i = 1#4300
        self.cam = cv2.VideoCapture('/home/runar/Ladybug/output0.mp4')
        #self.cam.set(1, 17000)
        
        
        self.image_time = str(sorted_file_list[i])
        self.imageNametoTimestamp(self.image_time)
        '''
        
        while not rospy.is_shutdown():
            corner = []
            corners = None
            if self.newimage == True:
                self.newimage = False
                self.looking_angle = np.deg2rad(0+self.Mounting_angle)*self.number
                if self.looking_angle > np.pi:
                    self.looking_angle -= 2*np.pi
                elif self.looking_angle < -np.pi:
                    self.looking_angle += 2*np.pi
                camera_mounting_offset = 3
                phi, theta, psi = self.euler_angles
                self.rotate_along_axis(self.image, -phi, -theta, -self.looking_angle)
                
                h = self.height
                w = self.width
                self.count+=1
                if self.count%5 == 0:#self.detect_ready:
                    self.yolo_image = self.warp.copy()
                    self.index +=1
                    if self.index > 2:
                        self.index = 0
                    i = self.index
                    #if self.index == 3:
                    #    im = self.warp[0:(h//4)*3,0:w]
                    #    tile = cv2.resize(im,(w//3,h//3))
                    #else:
                    tile = self.warp[h//3:(h//3)*2,(w//3)*i:(w//3)*i+(w//3)]

                    #img = cv2.imread('/home/runar/boat_single.jpg')
                    
                    self.yolopose = self.warppose
                    #corners = self.call_server(tile)

                    if corners is not None:
                        corners.sort(reverse = True, key=lambda x :x[1])
                        for c in corners:
                            if corner == []:
                                corner = [np.array(c[2])]
                            else:
                                corner.append(np.array(c[2]))
                        '''
                        cornerpose = self.rotation_3D(self.yolopose[0]-self.euler_angles[0], self.yolopose[1]-self.euler_angles[1], self.yolopose[2]-self.euler_angles[2])
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
                        corner = [xymin[0],xymin[1], xymax[0],xymax[1]]
                        print(corner)
                        '''
                        #print('Detected boat: ',corner)
                        self.show_webcam(self.yolo_image, corner)
                    else:
                        self.show_webcam(self.warp)

                else:
                    self.show_webcam(self.warp)
                    
                for d in self.detections:
                    cv2.line(self.draw, (d+int(w/2), 0), (d+int(w/2), h), (255,0,0), 10)
                self.detections = []
                #self.looking_angle
                for s in range(-8, 9):
                    mat2 = self.get_M(0.05, 0.1, s*np.pi/8, 0,0,0,0,0,1)
                    print(mat2, s*np.pi/8)
                mat2 = self.get_M(phi, theta, self.looking_angle, 0,0,0,0,0,1)
                #ime = np.eye(2)
                #im = np.array([1,1])
                #print(ime, im)
                #im2 = cv2.warpPerspective(im, mat2, (2, 2))
                rx = np.dot(mat2,[1,0,0])[1]
                tx = np.dot(mat2,[0,0,1])[1]
                rot = phi*np.cos(self.looking_angle)+theta*np.sin(self.looking_angle)
                trans = -phi*np.sin(self.looking_angle)+theta*np.cos(self.looking_angle)
                print(mat2,self.looking_angle)
                print(rx,phi,rot)
                print(tx,theta,trans)


                #print(im2)
                #print(np.dot(mat2,[0,h/2,self.looking_angle]))
                #print(np.dot(mat2,[w,h/2,self.looking_angle]))
                #cv2.line(self.draw, (int(p01[0]),int(p01[1])), (int(p02[0]),int(p02[1])), (0,255,0),2)
                '''
                print(np.dot(mat2,[0,1,0]))
                print(np.dot(mat2,[1,1,0]))
                print(np.dot(mat2,[0,0,1]))
                print(np.dot(mat2,[0,1,1]))
                print(np.dot(mat2,[1,0,0]))
                print(np.dot(mat2,[1,0,1]))
                '''
                #p1 = np.dot(mat2,[-1,0,0])
                #p2 = np.dot(mat2,[1,0,0])
                p1 = [-1,]
                #p1 = np.dot(mat2,[tx, ty, self.looking_angle+np.deg2rad(-50+camera_mounting_offset)])
                #p2 = np.dot(mat2,[np.deg2rad(-60+camera_mounting_offset),ty, x])
                #p2 = np.dot(mat2,[tx, ty, self.looking_angle+np.deg2rad(50+camera_mounting_offset)])
                #p4 = np.dot(mat2,[np.deg2rad(60+camera_mounting_offset),ty, x])
                #p5 = np.dot(mat2,[np.deg2rad(120+camera_mounting_offset),ty, x])
                #p3 = np.dot(mat2,[-ty,-tx, np.deg2rad(90+camera_mounting_offset)])
                #p4 = np.dot(mat2,[tx,-ty, np.deg2rad(180+camera_mounting_offset)])
                #print(p1,p2)
                p1 = self.angle_2_pixel(p1[0],p1[1],p1[2])
                p2 = self.angle_2_pixel(p2[0],p2[1],p2[2])
                #p3 = self.angle_2_pixel(p3[0],p3[1],p3[2])
                #p4 = self.angle_2_pixel(p4[0],p4[1],p4[2])
                #p5 = self.angle_2_pixel(p5[0],p5[1],p5[2])
                #p6 = self.angle_2_pixel(p6[0],p6[1],p6[2])
                #print(p1,p2)
                
                cv2.line(self.draw, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,255,0),2)
                #cv2.line(self.draw, (int(p2[0]),int(p2[1])), (int(p3[0]),int(p3[1])), (0,255,0),2)
                #cv2.line(self.draw, (int(p3[0]),int(p3[1])), (int(p4[0]),int(p4[1])), (0,255,0),2)
                #cv2.line(self.draw, (int(p4[0]),int(p4[1])), (int(p1[0]),int(p1[1])), (0,255,0),2)
                #cv2.line(self.draw, (int(p5[0]),int(p5[1])), (int(p6[0]),int(p6[1])), (0,255,0),2)
                #cv2.line(self.draw, (int(p6[0]),int(p6[1])), (int(p1[0]),int(p1[1])), (0,255,0),2)
                
                #self.data_assosiation()
                cv2.imshow('Cam3', self.draw)
                cv2.waitKey(1)
                self.count += 1



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