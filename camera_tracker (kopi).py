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
from darknet_ros_msgs.msg import CheckForObjectsAction, CheckForObjectsGoal, CheckForObjectsResult, CheckForObjectsActionFeedback


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
    def __init__(self,initialState, numSensors, numStates, sensorNoise, stateTransitionFunction, sensorTransferFunction, processNoise, processNoiseCovariance):
        self.numStates = numStates
        self.estimate = initialState  #current estimate, initialized with first observation
        self.previousEstimate = initialState  #previous state's estimate, initialized with first observation
        self.gain = np.array([[0,0],[0,0],[0,0],[0,0]]) #tradeoff of system between estimation and observation, initialized arbitrarily at identity
        self.previousGain = np.array([[0,0],[0,0],[0,0],[0,0]])  #previous gain, again arbitarily initialized
        self.errorPrediction = np.identity(numStates) #current estimation of the signal error,... starts as identity arbitrarily
        self.previousErrorPrediction = np.identity(numStates)  #previous signal error, again arbitrary initialization
        self.R = sensorNoise #variance of sensor noise
        self.f = stateTransitionFunction #state-transition function, from user input
        self.fJac = nd.Jacobian(self.f) #jacobian of f
        self.h = sensorTransferFunction  #sensor transfer function, from user input
        self.hJac = nd.Jacobian(self.h) #jacobian of h
        #self.processNoise = processNoise;  #process noise
        self.Q = processNoiseCovariance #process noise covariance

    def predict(self):
        """
        Called first.
        Predicts estimate and error prediction according to model of the situation
        """
        #update current state
        #self.estimate = np.dot(self.f, self.previousEstimate) + self.processNoise
        self.estimate = np.dot(self.f, self.previousEstimate)
        self.range = np.sqrt(self.estimate[0]**2+self.estimate[1]**2)
        #print('Range',self.range)

        #find current jacobian value
        #jacVal = self.fJac(self.previousEstimate) + self.processNoise

        #update error prediction state
        #self.errorPrediction = np.dot(jacVal , np.dot(self.previousErrorPrediction,np.transpose(jacVal))) + self.Q
        self.errorPrediction = np.dot(self.f , np.dot(self.previousErrorPrediction,np.transpose(self.f))) + self.Q

    def update(self,currentObservation=None):
        #currentObservation = np.array([[0.1],[1.0]])
        #currentObservation = np.transpose(y)
        """
        Called second.
        Updates estimate according to combination of observed and prediction.
        Also updates our learning parameters of gain and errorprediction.
        """

        #if all(self.estimate) == 0 or (abs(self.estimate[0]-currentObservation[0] < 200) and abs(self.estimate[1]-currentObservation[1] < 200)):
            #update the current estimate based on the gain
        self.estimate = self.estimate + np.dot(self.gain,(currentObservation - np.dot(self.h, self.estimate)))
        #find current jacobian value
        #jacVal = np.dot(self.hJac, self.estimate)

        #update the gain based on results from hte previous attempt at estimating
        #invVal = np.dot(jacVal, np.dot(self.errorPrediction, np.transpose(jacVal))) + self.R
        #self.gain = np.dot(self.errorPrediction, np.dot(np.transpose(jacVal) , np.linalg.inv(invVal) ))
        invVal = np.dot(self.h, np.dot(self.errorPrediction, np.transpose(self.h))) + self.R
        self.gain = np.dot(self.errorPrediction, np.dot(np.transpose(self.h) , np.linalg.inv(invVal) ))
        
        #update error prediction based on our success
        #self.errorPrediction = np.dot((np.identity(self.numStates) - np.dot(self.gain, jacVal)), self.errorPrediction)
        self.errorPrediction = np.dot((np.identity(self.numStates) - np.dot(self.gain, self.h)), self.errorPrediction)

        #update variables for next round
        self.previousEstimate = self.estimate
        self.previousGain = self.gain
        self.previousErrorPrediction = self.errorPrediction;
        #print(self.gain,self.errorPrediction)
        #print('prev',self.previousEstimate)
        

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
        self.target = {}
        self.kf = {}
        self.new_radar = []
        self.new_camera = False
        #self.myEstimate = [[0],[0],[0],[0]]

        self.templates = {}
        self.temp_num = 0
        self.corners = None
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
        #self.image3 = []
        #self.radar_img = []
        self.dark_id = None
        self.prev_dark_id = None
        self.dark_stamp = 0
        self.prev_dark_stamp = None
        self.millisecond = None
        self.firstimage = None
        self.second = None
        self.minute = None
        self.hour = None
        self.newimagetimestamp = 0
        self.radar_pixel = 0
        self.detections = []
        self.radar_detections = {}
        self.bb_angles = []
        self.range = 100
        self.track_id = None
        
        self.focal = 1350
        self.number = 0
        self.Mounting_angle = 72       # 5 cameras, 360/5=72
        self.fov_radians = np.deg2rad(100)      #FOV is about 100 deg
        self.fov_pixel = self.fov_radians/1616#self.width

        self.matrix = None
        self.penalty = np.zeros([100])

        # Estimation parameter of EKF
        dT = 0.01    #Timestep
        sigma_cv = 0.5    #Process noise strength
        sigma_l = 0.5   #Sensor noise strength
        self.processNoise = np.diag([1.0, 1.0, 1.0, 1.0])**2  # predict state covariance
        self.sensorNoise = sigma_l**2 * np.eye(2) #diag([10., 10.])**2  # Observation x,y position covariance
        self.processNoiseCovariance = sigma_cv**2 * np.array([[(dT**4)/4,0, (dT**3)/2,0],
                                                             [0, (dT**4)/4, 0,(dT**3)/2],
                                                             [(dT**3)/2, 0, (dT**2),  0],
                                                             [0, (dT**3)/2, 0,  (dT**2)]])
        self.initialReading = np.array([[0],[0],[0],[0]])
        self.stateTransfer =   np.array([[1,0,dT,0],
                                [0,1,0,dT],
                                [0,0,1,0],
                                [0,0,0,1]])
        self.sensorTransfer =  np.array([[1,0,0,0],
                                [0,1,0,0]])
        self.myEstimate = {}

    def EKF_init(self, i=None):
        #EKF_node = ExtendedKalman()   
        #EKF_node.start()
        if i is not None:
            self.kf[i] = ExtendedKalman(self.initialReading,2,4,self.sensorNoise,self.stateTransfer,self.sensorTransfer, self.processNoise, self.processNoiseCovariance)


    def extended_kalman(self):
        #print('EKF: ',msg)
        phi, theta, psi = self.euler_angles
        y = None
        radar_detections2 = self.radar_detections.copy()
        for i in radar_detections2:
            if i not in self.kf:
                self.EKF_init(i)
                self.kf[i].predict()
                self.myEstimate[i] = self.kf[i].getEstimate()
        for i in self.target:
            if self.target[i] > 1000:
                del self.kf[i]
                self.target[i] = -1

        for i in self.kf:
            self.kf[i].predict()
            
            for j in radar_detections2:
                if i == j:
                    y = radar_detections2[j]
                    break
            else:
                est_x = self.myEstimate[i][0]
                est_y = self.myEstimate[i][1]
                #print('xy',est_x,est_y)
                est_range = np.sqrt(est_x**2+est_y**2)
                
                if est_range < 1000:
                    
                    angle_ned = np.arctan2(est_x,est_y)
                    angle_body = angle_ned - psi + np.deg2rad(-3.14)     # Installation angle offset between radar and vessel  
                    #print('AAAAA',angle_body,self.bb_angles)
                    if angle_body < -np.pi:
                        angle_body += 2*np.pi
                    elif angle_body > np.pi:
                        angle_body -= 2*np.pi  
                    
                    a = []
                    for k in self.bb_angles:
                        a.append([abs(angle_body - k), k])
                    #print('bbb',b)
                    if a != []:
                        #b = min(abs(a[0]))
                        b = min(a, key=lambda x: x[0])
                        dx = float(est_range*np.cos(b[1]))
                        dy = float(est_range*np.sin(b[1]))
                        y = np.array([[dx], [dy]])
                        
            #print('breaked???',y)
            if y is not None:
                print('Y    ',y)
                self.kf[i].update(y)
            #grab result for this iteration 
            self.myEstimate[i] = self.kf[i].getEstimate()
            #print(i,len(self.kf), self.myEstimate[i])
            self.target[i] += 1
            '''
            for d in self.detections:
                dx = self.posterior_pos.x - self.position.x
                dy = self.posterior_pos.y - self.position.y
                phi, theta, psi = self.euler_angles

                self.radar_range = np.sqrt(dx**2+dy**2)

                radar_angle_ned = np.arctan2(dx,dy)
                self.radar_angle_body = radar_angle_ned - psi + np.deg2rad(-3.14)     # Installation angle offset between radar and vessel frame 
                if self.radar_angle_body < -np.pi:
                    self.radar_angle_body += 2*np.pi
                elif self.radar_angle_body > np.pi:
                    self.radar_angle_body -= 2*np.pi

                y = np.array([[dx],[dy]])
                #predict & update
                self.kf[i].predict()
                self.kf[i].update(y)
            '''    
                
        self.radar_detections = {}
        self.bb_angles = []
        '''
        elif self.new_camera:
            y = np.array([[0],[0]])
            self.new_camera = False
        #predict & update
        self.kf.predict()
        if y is None:
            self.kf.update()
        else:
            self.kf.update(y)

        #grab result for this iteration 
        myEstimate = self.kf.getEstimate()
        print(myEstimate)

        for i in self.new_radar:
            

        
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
        try:
            if self.radar_detections != []:
                for d in self.radar_detections:
                    for ang in self.bb_angles:
                        #print('A')
                        a.append([abs(int(d[0])-ang), abs(int(d[1])-self.range), d[2], d])
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
                    #self.extended_kalman(e[0][3])
                except:
                    print('data assosiation failed')
                
        #else:
         #   print('No radar detections')
        except:
            print('Radar assosiation error')
        #print('NOE')

        

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
        #radar_stamp = float(str(msg.header.stamp))/1e9
        #print(self.radar_stamp, self.pose_stamp)
        radar_track_id = msg.track_id
        posterior_pos = msg.posterior.pos_est
        #posterior_vel = msg.posterior.vel_est
        #posterior_pos_cov = msg.posterior.pos_cov
        #posterior_vel_cov = msg.posterior.vel_cov
        #print(self.posterior_pos, self.posterior_vel)
        #self.posterior_ned = np.array([posterior_pos.x, posterior_pos.y])
        #print(self.radar_stamp)
        #print(self.centroid_ned, self.ned)
        dx = posterior_pos.x - self.position.x
        dy = posterior_pos.y - self.position.y
        phi, theta, psi = self.euler_angles

        radar_angle_ned = np.arctan2(dx,dy)
        radar_angle_body = radar_angle_ned - psi + np.deg2rad(-3.14)     # Installation angle offset between camera and radar  

        if radar_angle_body < -np.pi:
            radar_angle_body += 2*np.pi
        elif radar_angle_body > np.pi:
            radar_angle_body -= 2*np.pi

        radar_angle_image = radar_angle_body - self.number*np.deg2rad(self.Mounting_angle)
        radar_pixel = int(radar_angle_image/self.fov_pixel)
        #radar_range = np.sqrt(dx**2+dy**2)
        self.radar_detections[radar_track_id] = np.array([[dx], [dy]]) 
        #self.radar_detections[self.radar_track_id] = (np.array([self.radar_angle_body, self.radar_range, self.radar_track_id,
        #    [self.posterior_vel.x, self.posterior_vel.y], [self.posterior_pos_cov.var_x, self.posterior_pos_cov.var_y, self.posterior_pos_cov.cor_xy], 
        #    [self.posterior_vel_cov.var_x, self.posterior_vel_cov.var_y, self.posterior_vel_cov.cor_xy]]))
        self.detections.append(radar_pixel)

        self.target[radar_track_id] = 0
        #cv2.line(self.warp, (self.radar_pixel, 0), (self.radar_pixel, self.height), (0,255,0), 10)
        #print('RADAR_ANGLE',radar_angle_ned, psi, self.radar_angle_body, self.radar_angle_image, self.radar_pixel)
        

    def feedback_callback(self, msg):
        print('DARK FEEDBACK',msg)
        #result = self.dark_client.get_result()
    def darknet_callback(self, status, result):
        #print('DARKNET feedback:  ', status, result.id)
        corners = None
        #corner = []
        h = self.height
        w = self.width
        if status == 3:
            self.dark_stamp = float(str(result.bounding_boxes.header.stamp))/1e9
            i = self.dark_id = int(result.id)
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

                self.corners = corners
            self.detect_ready = True
                #print(i,corners)

    def call_server(self,image, index):
        #self.dark_client.wait_for_server()
        bridge = CvBridge()
        im = bridge.cv2_to_imgmsg(image, 'bgr8')#encoding="passthrough")
        goal = CheckForObjectsGoal()
        goal.id = index
        goal.image = im
        self.dark_client.send_goal(goal, done_cb = self.darknet_callback, feedback_cb=self.feedback_callback)
        #self.dark_client.wait_for_result()
        #result = self.dark_client.get_result()

        #return self.corners


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
        
    def re3_track(self, image, corners=None):   
        global initialize, boxToDraw#,tracker

        if corners is None:
            try:
                boxToDraw = tracker.track(image[:,:,::-1], 'Cam')
            except:
                print("No Bbox to track")

        if boxToDraw is not None and corners is not None:
            iou = bb_intersection_over_union(boxToDraw,corners)
            if iou < 0.3:
                initialize = True
                print ("UPDATED")

        if initialize and corners is not None:
            if 0 not in corners:
                boxToDraw = corners
                initialize = False
                boxToDraw = tracker.track(image[:,:,::-1], 'Cam', boxToDraw)


        if boxToDraw is not None:
            if any(boxToDraw) != 0:
                bb_angles = []
                #for a in boxToDraw:
                b = boxToDraw
                #print(a,b)
                if ((abs(b[0]-b[2]) > 4) and (abs(b[1]-b[3]) > 4)):
                    cv2.rectangle(self.draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), 
                        [0,0,255], 2)
                    bb_angle = (self.fov_pixel*(int(b[0])+(int(b[2])-int(b[0]))/2-self.width/2)+self.number*np.deg2rad(self.Mounting_angle))
                    print(bb_angle)
                    self.bb_angles.append(bb_angle)
                self.new_camera = True


    def re3_multi_track(self, image, corners=None):
        box = {}
        #i = 0
        h = self.height
        w = self.width
        global initialize, boxToDraw#,tracker

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
            #print(len(boxToDraw),len(corners))
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
            #print(indexes)
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
            #bboxes = tracker.track(image[:,:,::-1], boxToDraw.keys(), box)
            #print('From tracker',bboxes)
            i = 0
            for b in boxToDraw:
                boxToDraw[b] = bboxes[i]
                i += 1

            self.corners = None

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
                #bboxes = tracker.track(image[:,:,::-1], box.keys())
                i = 0
                for b in box:
                    boxToDraw[b] = bboxes[i]
                    i += 1
            except:
                print("No Bbox to track")
        if boxToDraw is not None:
            if any(boxToDraw) != 0:
                bb_angles = []
                for a in boxToDraw:
                    b = boxToDraw[a]
                    #print(a,b)
                    if ((abs(b[0]-b[2]) > 4) and (abs(b[1]-b[3]) > 4)):
                        cv2.rectangle(self.draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), 
                            [0,0,255], 2)
                        bb_angle = (self.fov_pixel*(int(b[0])+(int(b[2])-int(b[0]))/2-self.width/2)+self.number*np.deg2rad(self.Mounting_angle))
                        print(bb_angle)
                        self.bb_angles.append(bb_angle)
                self.new_camera = True
                

                    
    def template_tracker(self, image, corners=None):
        #print(corners)
        im = image[self.height/3:2*self.height/3,:]
        
        if corners is not None:
            for c in corners:
                if self.templates == {}:
                    self.templates[0] = [c,image[0:2,0:2]]
                else:
                    for tname in self.templates:
                        templ = self.templates[tname][1]
                        #print(templ.shape,c)
                        h, w = templ.shape[:2]
                        res = cv2.matchTemplate(im,templ,cv2.TM_CCOEFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                        top_left = max_loc
                        bottom_right = (top_left[0] + h, top_left[1] + w)
                        
                        print(top_left,bottom_right)
                        t = self.templates[tname][0]
                        if c[0] > t[0]-50 and c[1] > t[1]-50 and c[2] < t[2]+50 and c[3] < t[3]+50:
                            self.templates[tname] = [c, templ]
                            self.penalty[tname] = 0
                            cv2.rectangle(im,top_left, bottom_right, [0,255,255], 2)
                        else:
                            if self.penalty[tname] > 500:
                                del self.templates[tname]
                                break
                            else:
                                self.penalty[tname] +=1
                                self.temp_num += 1
                                self.templates[self.temp_num] = [c, image[int(c[1]):int(c[3]),int(c[0]):int(c[2])]]
                                cv2.rectangle(im,top_left, bottom_right, [255,0,255], 2)
                                break
                        print(len(self.templates))
                    cv2.imshow('im',im)
                    cv2.waitKey(1)

                        
            
    def pixel_2_angle(self, x, y, z=0):
        x_angle = x*self.fov_pixel-(self.width/2)*self.fov_pixel+np.deg2rad(self.Mounting_angle)
        y_angle = y*self.fov_pixel-(self.height/2)*self.fov_pixel
        z_angle = z
        return [x_angle, y_angle, x_angle] 

    def angle_2_pixel(self, r):
        return (r/self.fov_pixel)#+(self.width/2)#-np.deg2rad(self.Mounting_angle*self.fov_pixel)
        #y = (ry/self.fov_pixel)#+(self.height/2)
        #z = 1
        #return [x, y, z]
        
    def rotate_along_axis(self, image, phi=0, theta=0, psi=0, dx=0, dy=0, dz=0):
        self.warppose = self.euler_angles
        # Get ideal focal length on z axis
        dz = self.focal*1.
        #axis = np.float32([[3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)

        # Get projection matrix
        self.mat = self.get_M(phi, theta, psi, dx, dy, dz)

        self.warp = cv2.warpPerspective(image, self.mat, (self.width, self.height))
        self.draw = self.warp.copy()
        #print('WARP')
        

    """ Get Perspective Projection Matrix """
    def get_M(self, phi, theta, psi, dx=0, dy=0, dz=0, w=None, h=None, f=None):

        if w is None:
            w=self.width
            h=self.height
            f=self.focal

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
        
        R = self.rotation_3D(phi, theta, psi)
        #R1 = self.rotation_3D(theta,0,0)    # Center horizon in image
        R2 = self.rotation_3D(0,-psi,0)     # Rotate image back onto imageplane

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
        return np.dot(A2, np.dot(T, np.dot(R2,np.dot(RCB, A1))))
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
        return np.dot(RX, np.dot(RY, RZ))
        #print(R,RX,RY,RZ)

        

    def start(self):
        self.EKF_init()
        self.rate = rospy.Rate(100)
        # Subscribers
        rospy.Subscriber('/seapath/pose',geomsg.PoseStamped, self.pose_callback)
        rospy.Subscriber('/radar/estimates', automsg.RadarEstimate, self.radar_callback)
        rospy.Subscriber('/ladybug/camera0/image_raw', Image, self.image_callback)

        self.dark_client = actionlib.SimpleActionClient('darknet_ros/check_for_objects', darknetmsg.CheckForObjectsAction)

        while not rospy.is_shutdown():
            corner = []
            self.extended_kalman()

            if self.newimage == True:
                self.newimage = False
                self.looking_angle = np.deg2rad(0+self.Mounting_angle)*self.number
                if self.looking_angle > np.pi:
                    self.looking_angle -= 2*np.pi
                elif self.looking_angle < -np.pi:
                    self.looking_angle += 2*np.pi
                camera_mounting_offset = 3
                phi, theta, psi = self.euler_angles
                self.rotate_along_axis(self.image, -phi, -theta, -self.looking_angle,
                    0,-self.angle_2_pixel(theta)*np.cos(self.looking_angle)+self.angle_2_pixel(phi)*np.sin(self.looking_angle),0)

                h = self.height
                w = self.width
                self.count+=1
                    
                if self.detect_ready or float(str(rospy.Time.now()))/1e9-self.dark_stamp > 20:
                    if self.dark_id != self.prev_dark_id or self.prev_dark_id is None:
                        self.prev_dark_id = self.dark_id
                        self.detect_ready = False
                        self.index +=1
                        if self.index > 2:
                            self.index = 0
                    
                    i = self.index
                    self.yolo_image = self.warp.copy()

                    tile = self.warp[h//3:(h//3)*2,(w//3)*i:(w//3)*i+(w//3)]

                    self.call_server(tile, i)

                if self.corners is not None:
                    self.corners.sort(reverse = True, key=lambda x :x[1])
                    for c in self.corners:
                        if corner == []:
                            #print(c)
                            if c[2][3] < self.height/2+50:
                                corner = [np.array(c[2])]
                        else:
                            if c[2][3] < self.height/2+50:
                                corner.append(np.array(c[2]))
                    
                    if corner == []:
                        #self.template_tracker(self.warp)
                        #self.re3_track(self.warp)
                        self.re3_multi_track(self.warp)
                    else:
                        #c = []
                        #c.append(corner[0])
                        #self.template_tracker(self.yolo_image, c)
                        #corner = corner[0]
                        #self.re3_track(self.yolo_image, corner)
                        self.re3_multi_track(self.yolo_image, corner)
                else:
                    #self.template_tracker(self.warp)
                    #self.re3_track(self.warp)
                    self.re3_multi_track(self.warp)

                    
                for d in self.detections:
                    cv2.line(self.draw, (d+int(w/2), 0), (d+int(w/2), h), (255,0,0), 10)
                self.detections = []

                cv2.imshow('Cam3', self.draw)
                cv2.waitKey(1)
                #self.count += 1

                '''
                [[ 0.92388   0.369644  0.099046]
                 [-0.382683  0.892399  0.239118]
                 [ 0.       -0.258819  0.965926]]   # 0 deg

                [[ 0.099046  0.369644 -0.92388 ]
                 [ 0.239118  0.892399  0.382683]
                 [ 0.965926 -0.258819  0.      ]]   # 90 deg
                '''
            self.rate.sleep()



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