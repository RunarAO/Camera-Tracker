#!/usr/bin/env python
import cv2
import argparse
#import glob
#from skimage import transform as sk_transform
import numpy as np
from numpy.linalg import norm
from scipy.optimize import fmin
#from scipy.spatial.transform import Rotation as R
import rospy
import os
import time
import sys
import getopt
import math
import struct
import random
from cv_bridge import CvBridge, CvBridgeError
#import tf
#from tf import TransformListener
#import message_filters
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from munkres import Munkres, print_matrix, make_cost_matrix
import numdifftools as nd
import numdifftools.nd_algopy as nda

'''
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
'''

import autosea_msgs.msg as automsg
import std_msgs.msg as stdmsg
import geometry_msgs.msg as geomsg
import visualization_msgs.msg as vismsg
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


np.set_printoptions(precision=10)
np.set_printoptions(suppress=True)


boxToDraw = None#np.zeros(4)
initialize = True
#number = None



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
    def __init__(self,initialState, initialCovariance, numSensors, numStates, sensorNoise, stateTransitionFunction, sensor_1_TransferFunction, processNoiseCovariance):
        self.numStates = numStates
        self.estimate = initialState  #current estimate, initialized with first observation
        self.previousEstimate = initialState  #previous state's estimate, initialized with first observation
        self.gain = np.array([[1,1,1],[1,1,1],[1,1,1],[1,1,1]]) #tradeoff of system between estimation and observation, initialized arbitrarily at identity
        self.previousGain = np.array([[1,1,1],[1,1,1],[1,1,1],[1,1,1]])  #previous gain, again arbitarily initialized
        self.errorPrediction = initialCovariance #current estimation of the signal error,... starts as identity arbitrarily
        self.previousErrorPrediction = initialCovariance #99999*np.identity(numStates)  #previous signal error, again arbitrary initialization
        self.R = sensorNoise #variance of sensor noise
        self.f = stateTransitionFunction #state-transition function, from user input
        self.fJac = nd.Jacobian(self.f) #jacobian of f
        self.h1 = sensor_1_TransferFunction  #sensor transfer function, from user input
        #self.h2 = sensor_2_TransferFunction
        self.hJac = nd.Jacobian(self.h1) #jacobian of h
        #self.processNoise = processNoise;  #process noise
        self.Q = processNoiseCovariance #process noise covariance

    def predict(self):
        """
        Called first.
        Predicts estimate and error prediction according to model of the situation
        """
        #update current state
        self.estimate = self.f(self.previousEstimate)
        #self.estimate = np.dot(self.f, self.previousEstimate)# + self.processNoise
        
        #self.range = np.sqrt(self.estimate[0]**2+self.estimate[1]**2)

        #find current jacobian value
        jacVal = self.fJac(self.previousEstimate) #+ self.Q
        #print('estimate: ',jacVal)

        #update error prediction state
        self.errorPrediction = np.dot(jacVal , np.dot(self.previousErrorPrediction,np.transpose(jacVal))) + self.Q
        #self.errorPrediction = np.dot(self.f , np.dot(self.previousErrorPrediction,np.transpose(self.f))) + self.Q

    def update(self,currentObservation=None, sensorNoise=None, only_angle=False):
        if sensorNoise is not None:
            self.R = sensorNoise
        if currentObservation is not None:
            if only_angle == True:
                #print(currentObservation)
                gain = self.gain
                #gain[:,[0,1]] = 0
            else:
                gain = self.gain
                #gain[:,[2]] = 0
            #self.estimate = self.estimate
        #else:
            '''
            if len(shape(currentObservation)) > 1:
                self.estimate = self.estimate + np.dot(self.gain,(currentObservation - np.dot(self.h2,self.estimate)))
                invVal = np.dot(self.h, np.dot(self.errorPrediction, np.transpose(self.h))) + self.R
                self.gain = np.dot(self.errorPrediction, np.dot(np.transpose(self.h) , np.linalg.inv(invVal) ))
                self.errorPrediction = np.dot((np.identity(self.numStates) - np.dot(self.gain, self.h)), self.errorPrediction)
            else:
            '''
            #print(gain, currentObservation,self.estimate)
            #print(currentObservation - self.h1(self.estimate))
            #print(np.dot(self.gain,(currentObservation - self.h1(self.estimate))))
            self.estimate = np.add(self.estimate, (np.dot(gain,(currentObservation - self.h1(self.estimate)))).reshape(-1,1))
            jacVal = self.hJac(self.estimate)
            invVal = np.dot(jacVal, np.dot(self.errorPrediction, np.transpose(jacVal))) + self.R
            self.gain = np.dot(self.errorPrediction, np.dot(np.transpose(jacVal) , np.linalg.inv(invVal) ))
            self.errorPrediction = np.dot((np.identity(self.numStates) - np.dot(self.gain, jacVal)), self.errorPrediction)
            '''
            print(self.estimate)
            print(self.gain)
            print(self.errorPrediction)
            '''
        """
        Called second.
        Updates estimate according to combination of observed and prediction.
        Also updates our learning parameters of gain and errorprediction.
        """
        '''
        #update the current estimate based on the gain
        if currentObservation is None:
            self.estimate = self.estimate
        else:
            self.estimate = self.estimate + np.dot(self.gain,(currentObservation - self.h(self.estimate)))
            #self.estimate = self.estimate + np.dot(self.gain,(currentObservation - np.dot(self.h,self.estimate)))
        
        #find current jacobian value
        jacVal = self.hJac(self.estimate)#self.estimate[0])
        #print('update: ',jacVal)
        
        #update the gain based on results from hte previous attempt at estimating
        invVal = np.dot(jacVal, np.dot(self.errorPrediction, np.transpose(jacVal))) + self.R
        self.gain = np.dot(self.errorPrediction, np.dot(np.transpose(jacVal) , np.linalg.inv(invVal) ))
        #invVal = np.dot(self.h, np.dot(self.errorPrediction, np.transpose(self.h))) + self.R
        #self.gain = np.dot(self.errorPrediction, np.dot(np.transpose(self.h) , np.linalg.inv(invVal) ))
        
        #update error prediction based on our success
        self.errorPrediction = np.dot((np.identity(self.numStates) - np.dot(self.gain, jacVal)), self.errorPrediction)
        #self.errorPrediction = np.dot((np.identity(self.numStates) - np.dot(self.gain, self.h)), self.errorPrediction)
        '''
        #update variables for next round
        self.previousEstimate = self.estimate
        self.previousGain = self.gain
        self.previousErrorPrediction = self.errorPrediction
        #print(self.gain,self.errorPrediction)
        #print('prev',self.previousEstimate)
        

    def getEstimate(self):
        """
        Simple getter for cleanliness
        """
        return self.estimate

    def getGain(self):
        return self.gain

    def getP(self):
        return self.errorPrediction




class DetectObjects(object):
    def __init__(self):
        # Params
        self.target = {}
        self.kf = {}
        self.new_radar = []
        self.new_camera = False
        #self.myEstimate = [[0],[0],[0],[0]]
        self.newcameramtx = None
        self.templates = {}
        self.temp_num = 0
        self.corners = None
        self.detect_ready = True
        self.count = 0
        self.imcount = 0
        self.index = 0
        #self.q_b_w = []
        self.pos = []
        #self.ned = []
        self.euler_angles = [0,0,0]
        #self.tile = []
        #self.tiles = []
        self.newimage = {}
        self.image = {}
        self.draw = None
        self.warp = {}
        #self.previmg = None
        #self.image3 = []
        #self.radar_img = []
        self.dark_id = None
        self.prev_dark_id = None
        self.dark_stamp = 0
        self.prev_dark_stamp = 0
        self.darkpose = [0,0,0]
        self.millisecond = None
        self.firstimage = None
        self.second = None
        self.minute = None
        self.hour = None
        self.newimagetimestamp = 0
        self.radar_pixel = 0
        self.detection = {}
        self.detections = {}
        self.c_detections = []
        self.radar_detections = {}
        self.ais_detections = {}
        self.angle_ned = 0
        self.posterior_pos_cov = {}
        self.bb_angles = []
        self.bb_anglesyolo = {}
        self.yolo_angle = {}
        self.range = 100
        #self.delta = {}
        self.track_id = None
        self.radar = {}
        #self.ry = {}
        self.ais = {}
        self.im_attitude = {}
        self.looking_angle = 0
        self.focal = 1640#1350
        self.number = None
        self.Mounting_angle = np.deg2rad(72)       # 5 cameras, 360/5=72
        self.camera_offset = np.deg2rad(-3.5)        # psi degrees between camera and vessel
        self.radar_offset = np.deg2rad(3)         # psi degrees between radar and vessel
        self.fov_radians_hor = np.deg2rad(106.8) #106.8 #92      #FOV is about 90-100 deg
        self.fov_radians_ver = np.deg2rad(70)      #FOV is about 110 deg
        self.fov_pixel_hor = self.fov_radians_hor/1616#self.width
        self.fov_pixel_ver = self.fov_pixel_hor-0.0001 #self.fov_radians_ver/1264#self.height ca

        self.matrix = None
        self.penalty_radar = {}#np.zeros([100])
        self.penalty_ais = {}
        self.penalty = {}

        # Estimation parameter of EKF
        self.updateRate = 50
        self.dT = 1./self.updateRate    #Timestep
        self.decay = 0.95
        sigma_cv = 0.45#100    #Process noise strength
        sigma_r = 0.4#5   #Sensor noise strength
        sigma_a = 0.4#0.5   #Sensor noise strength
        sigma_cd = 0.001#0.5   #Sensor noise strength
        sigma_ct = 0.005#0.5   #Sensor noise strength
        #self.processNoise = np.diag([1.0, 1.0, 1.0, 1.0])**2  # predict state covariance
        self.radarNoise =  sigma_r**2 * np.array([[1,0,0],
                                                  [0,1,0],
                                                  [0,0,0.11]]) #diag([10., 10.])**2  # Observation x,y position covariance
        self.aisNoise =  sigma_a**2 * np.array([[1,0,0],
                                                [0,1,0],
                                                [0,0,0.1]])
        self.cameraNoise_yolo = np.array([[sigma_cv**2,0,0],
                                            [0,sigma_cv**2,0],
                                            [0,0,sigma_cd**2]])
        self.cameraNoise_tracker = np.array([[2*sigma_cv**2,0,0],
                                               [0,2*sigma_cv**2,0],
                                               [0,0,sigma_ct**2]])
        '''self.processNoiseCovariance = sigma_cv**2 * np.array([[(self.dT**4)/4,0, (self.dT**3)/2,0,0,0],
                                                             [0, (self.dT**4)/4, 0,(self.dT**3)/2,0,0],
                                                             [(self.dT**3)/2, 0, (self.dT**2),  0,0,0],
                                                             [0, (self.dT**3)/2, 0,  (self.dT**2),0,0]
                                                             [0,0,0,0,0,0]
                                                             [0,0,0,0,0,0]])'''
        self.processNoiseCovariance = sigma_cv**2 * np.array([[(self.dT**4)/4,0, (self.dT**3)/2,0],
                                                             [0, (self.dT**4)/4, 0,(self.dT**3)/2],
                                                             [(self.dT**3)/2, 0, (self.dT**2),  0],
                                                             [0, (self.dT**3)/2, 0,  (self.dT**2)]])
        #self.initialReading = np.array([[0],[0],[0],[0]])
        self.stateTransfer =   np.array([[1,0,self.dT*self.decay,0],
                                         [0,1,0,self.dT*self.decay],
                                         [0,0,1,0,0,0],
                                         [0,0,0,1,0,0]])
        self.sensorTransfer =  np.array([[1,0,0,0],
                                         [0,1,0,0]])
        self.myEstimate = {}
        #print(self.processNoiseCovariance)
        self.linje = np.linspace(1815,2024,10000)
        #plt.plot(self.linje, 2.357142857*x -2072.21, '-k')
    
    def f(self, x):
        #dT = 0.02
        #print(str(x))
        x2 = float(x[2][0])
        x3 = float(x[3][0])
        x0 = float(x[0][0])+self.dT*x2#*self.decay
        x1 = float(x[1][0])+self.dT*x3#*self.decay
        #r = np.sqrt(x0**2+x1**2)
        #theta = np.arctan2(x2,x1)
        return np.array([[x0],[x1],[x2],[x3]])#,[r],[theta]])

    def h_radar(self, x):
        return np.array([[x0],[x1]])

    def h(self, x):
        #print(x)
        #r = float(x[4][0])
        #theta = float(x[5][0])
        x0 = float(x[0][0])
        x1 = float(x[1][0])
        dx0 = x0-self.position.x
        dx1 = x1-self.position.y
        #print(x0, x1)
        #x0 = float(r*np.cos(theta))
        #x1 = float(r*np.sin(theta))
        #return np.array([[x0],[x1]])
        return np.array([[x0], [x1], [np.arctan2(dx1, dx0)]])

    #def g(self, r, theta, x):
    #    dr = r-float(x[4][0])
    #    return np.array[]

    def EKF_init(self, i, x, cov):
        #EKF_node = ExtendedKalman()   
        #EKF_node.start()
        #if i is not None:
        initialReading = [x[0],x[1],x[2],x[3]]
        self.sensorNoise = self.radarNoise
        initialCovariance = [[(cov[0][0]),(cov[0][1]),0,0],
                             [(cov[1][0]),(cov[1][1]),0,0],
                             [0,0,25**2,0],
                             [0,0,0,25**2]]
        self.kf[i] = ExtendedKalman(initialReading,initialCovariance, 1,4,self.sensorNoise, self.f, self.h, self.processNoiseCovariance)

    def dist_to_point_comp(self, point):
        p1 = np.array([2206,1815])
        p2 = np.array([2701,2025])
        #p1 = np.array([1209,1419])
        #p2 = np.array([1422,1501])
        dx = p2[0]-p1[0]
        dy = p2[1]-p1[1]
        dr2 = float(dx ** 2 + dy ** 2)

        lerp = ((point[0] - p1[0]) * dx + (point[1] - p1[1]) * dy) / dr2
        if lerp < 0:
            lerp = 0
        elif lerp > 1:
            lerp = 1

        x = lerp * dx + p1[0]
        y = lerp * dy + p1[1]

        _dx = x - point[0]
        _dy = y - point[1]
        square_dist = _dx ** 2 + _dy ** 2
        return square_dist

    def dist_to_point(self, point):
        return math.sqrt(self.dist_to_point_comp(point))

    def extended_kalman(self):
        t = time.time()
        #print('EKF: ',msg)
        p1 = np.array([2206,1815])
        p2 = np.array([2701,2025])

        #phi, theta, psi = self.euler_angles
        radar_angle = None
        diff2 = 999
        y = None
        y1 = None
        R = self.radarNoise
        angle_only = False
        
        #b = None
        radar_detections2 = self.radar_detections.copy()
        ais_detections = self.ais_detections.copy()
        for i in radar_detections2:
            rx = radar_detections2[i][0]
            ry = radar_detections2[i][1]
            dx = rx-self.position.x
            dy = ry-self.position.y
            r_range = np.sqrt(dx**2+dy**2)
            if r_range < 2500:# or r_range > 1000:
                self.radar[i] = [rx,ry]
                self.penalty_radar[i] = 0
                self.penalty[i] = 0
            if i not in self.kf and r_range < 1000:
                self.EKF_init(i, radar_detections2[i], self.posterior_pos_cov[i])
                self.kf[i].predict()
                self.myEstimate[i] = self.kf[i].getEstimate()
                #self.delta[i] = np.array([0,0,0,0]).reshape(-1,1)
        for i in self.target:
            if self.target[i] > 100:
                del self.kf[i]
                self.target[i] = -1

        for i in self.kf:
            self.kf[i].predict()
            plot_ell = False
            est_x = self.myEstimate[i][0]
            est_y = self.myEstimate[i][1]
            est_dx = est_x-self.position.x
            est_dy = est_y-self.position.y
            est_xdot = self.myEstimate[i][2]
            est_ydot = self.myEstimate[i][3]
            #if abs(est_xdot) > 100 or abs(est_ydot) > 100:
            #    self.target[i] = 2000
            angle_est = np.arctan2(est_dy,est_dx)
            angle_body = angle_est - self.psi# + self.radar_offset #- self.looking_angle     # Installation angle offset between radar and vessel  
            
            #print('AAAAA',angle_body,self.bb_angles)
            if angle_est < -np.pi:
                angle_est += 2*np.pi
            elif angle_est > np.pi:
                angle_est -= 2*np.pi  

            if angle_body < -np.pi:
                angle_body += 2*np.pi
            elif angle_body > np.pi:
                angle_body -= 2*np.pi  
            
            #print('xy',est_x,est_y)
            est_range = np.sqrt(est_dx**2+est_dy**2)
            #print(psi, angle_ned, angle_body, est_range)
            
            
            for j in radar_detections2:
                if i == j and (est_range < 2500):# or est_range > 1000) :
                    x0 = radar_detections2[j][0]
                    x1 = radar_detections2[j][1]
                    radar_angle = np.arctan2(x1-self.position.y, x0-self.position.x)
                    #radar_range = np.sqrt((x1-self.position.y)**2 + (x0-self.position.x)**2)
                    y = np.array([x0,x1,radar_angle]).reshape(-1,1)
                    R = self.radarNoise
                    p3 = np.array([x0,x1])
                    #d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
                    d = self.dist_to_point(p3)
                    if d < 20:
                        #print(d)
                        plt.plot(float(x1), float(x0), 'or'
                        #plt.plot(float(self.count) ,float(d), 'or'
                            , label= 'Radar' if 'Radar' not in plt.gca().get_legend_handles_labels()[1] else '')
                    
                    #plt.plot(float(self.count), float(np.sqrt((self.myEstimate[i][0]-x0)**2+(self.myEstimate[i][1]-x1)**2)), 'or')
                    #g = self.kf[i].getGain()
                    #p = self.kf[i].getP()
                    #est = self.kf[i].getEstimate()
                    plot_ell = True
                    #break
            for a in ais_detections:
                aisx = ais_detections[a][0]
                aisy = ais_detections[a][1]
                if np.sqrt((est_x-aisx)**2+(est_y-aisy)**2) < 50:
                    ais_angle = np.arctan2(aisy-self.position.y, aisx-self.position.x)
                    #y1 = np.array([aisx,aisy,ais_angle]).reshape(-1,1)
                    #print(aisx, aisy)
                    R1 = self.aisNoise
                    p3 = np.array([aisx,aisy])
                    d = self.dist_to_point(p3)
                    if d < 20:
                        plt.plot(float(aisy), float(aisx), 'ob'
                        #plt.plot(float(self.count) ,float(d), 'ob'
                            , label= 'AIS' if 'AIS' not in plt.gca().get_legend_handles_labels()[1] else '')
                    self.ais[i] = [aisx,aisy]
                    self.penalty_ais[i] = 0
                    self.penalty[i] = 0
                    plot_ell = True
            else:
                if est_range < 1500:
                    a = []
                    yolo = False
                    re3 = False
                    for k in self.bb_anglesyolo:
                        yo = self.bb_anglesyolo[k]
                        if yo < -np.pi:
                            yo += 2*np.pi
                        elif yo > np.pi:
                            yo -= 2*np.pi 
                        #print('YO',yo)
                        #for yo in angle:
                        diff = angle_est-yo
                        if radar_angle is not None:
                            diff2 = radar_angle-yo
                        print('DIFF',diff,diff2)
                        
                        if abs(diff) < np.deg2rad(8) or abs(diff2) < np.deg2rad(8):
                            a.append([abs(diff), yo])
                            R = self.cameraNoise_yolo
                            yolo = True
                    else:
                        for k in self.bb_angles:
                            if k < -np.pi:
                                k += 2*np.pi
                            elif k > np.pi:
                                k -= 2*np.pi 
                            #print('re3',k)
                            diff = angle_est-k
                            if radar_angle is not None:
                                diff2 = abs(radar_angle-k)
                            if abs(diff) < np.deg2rad(4) or diff2 < np.deg2rad(4):
                                a.append([abs(diff), k])
                                R = self.cameraNoise_tracker
                                re3 = True
                    #print(a, angle_est, angle_body)
                    #if a != []:
                    if yolo or re3:
                        b = min(a, key=lambda x: x[0])
                        c = b[1] + self.camera_offset
                        if c < -np.pi:
                            c += 2*np.pi
                        elif c > np.pi:
                            c -= 2*np.pi 
                        #d = c - self.looking_angle - self.psi
                        #print(c)
                        dx = float(est_range*np.cos(c))
                        dy = float(est_range*np.sin(c))
                        p3 = np.array([float(dx+self.position.x),float(dy+self.position.y)])
                        d = self.dist_to_point(p3)

                        c_alpha = R[2][2]
                        if i in self.penalty:
                            y = np.array([float(dx+self.position.x) , float(dy+self.position.y) , c]).reshape(-1,1)
                            print('CAMERA',y)
                            self.penalty[i] += 1
                            R = R * self.penalty[i]**2
                            R[2][2] = c_alpha
                        
                        if i in self.radar:
                            y = np.array([float(self.radar[i][0]),float(self.radar[i][1]), c]).reshape(-1,1)
                            print('RADAR',y)
                            self.penalty_radar[i] += 1
                            R = np.minimum(R,self.radarNoise)*self.penalty_radar[i]
                            R[2][2] = c_alpha
                        elif i in self.ais:
                            y = np.array([self.ais[i][0], self.ais[i][1], c]).reshape(-1,1)
                            print('AIS',y)
                            self.penalty_ais[i] += 1
                            R = np.minimum(R,self.aisNoise)*self.penalty_ais[i]
                            R[2][2] = c_alpha
                        #R = self.cameraNoise 
                        #R = self.radarNoise
                        angle_only = True
                        
                        
                        

                        if yolo:
                            if d < 20:
                                plt.plot(float(dy+self.position.y), float(dx+self.position.x), 'og'
                                #plt.plot(float(self.count) ,float(d), 'og'
                                    , label= 'Camera_Detector' if 'Camera_Detector' not in plt.gca().get_legend_handles_labels()[1] else '')
                        else:
                            if d < 20:
                                plt.plot(float(dy+self.position.y), float(dx+self.position.x), '+g'
                                #plt.plot(float(self.count) ,float(d), '+g'
                                    , label= 'Camera_Tracker' if 'Camera_Tracker' not in plt.gca().get_legend_handles_labels()[1] else '')
                            
                        
            if est_range < 2500:
                #radar_angle_image = angle_body
            
                #radar_pixel = int(radar_angle_image/self.fov_pixel_hor)
                self.detections[i] = angle_body
            #print(y)
            self.kf[i].update(y, R, angle_only)
            if y1 is not None:
                self.kf[i].update(y1, R1, angle_only)
            y = None
            y1 = None
            #grab result for this iteration 
            #old = self.myEstimate[i]
            self.myEstimate[i] = self.kf[i].getEstimate()
            #self.delta[i] = np.subtract(self.myEstimate[i], old)
            p3 = np.array([self.myEstimate[i][0],self.myEstimate[i][1]])
            d = self.dist_to_point(p3)
            if d < 20:
                plt.plot(float(self.myEstimate[i][1]), float(self.myEstimate[i][0]), '+k'
                #plt.plot(float(self.count), d, '+k'
                    , label= 'Estimate' if 'Estimate' not in plt.gca().get_legend_handles_labels()[1] else '')
            
            plot_ell = False
            if plot_ell == True:
                nstd = 50
                ixgrid = np.ix_([0, 1], [0, 1])
                cov = self.kf[i].getP()
                cov = cov[ixgrid]
                vals, vecs = self.eigsorted(cov)
                theta = np.degrees((-90)+np.arctan2(*vecs[:,0][::-1]))
                w, h = 2 * nstd * np.sqrt(vals)
                ell = Ellipse(xy=(float(self.myEstimate[i][1]), float(self.myEstimate[i][0])),
                              width=w, height=h,
                              angle=theta, color='black')
                ell.set_facecolor('none')
                plt.axes().add_artist(ell)

                if i in self.radar:
                    nstd = 1
                    ixgrid = np.ix_([0, 1], [0, 1])
                    cov = np.array(self.posterior_pos_cov[i])
                    cov = cov[ixgrid]
                    vals, vecs = self.eigsorted(cov)
                    theta = np.degrees((-90)+np.arctan2(*vecs[:,0][::-1]))
                    w, h = 2 * nstd * np.sqrt(vals)
                    ell = Ellipse(xy=(float(self.radar[i][1]), float(self.radar[i][0])),
                                  width=w, height=h,
                                  angle=theta, color='red')
                    ell.set_facecolor('none')
                    plt.axes().add_artist(ell)

            if i in self.penalty_ais:
                if self.penalty_ais[i] > 20:
                    del self.ais[i]
                    del self.penalty_ais[i]
            if i in self.penalty_radar:
                if self.penalty_radar[i] > 20:    
                    del self.radar[i]
                    del self.penalty_radar[i]
        
        self.radar_detections = {}
        self.ais_detections = {}
        self.bb_angles = []
        self.bb_anglesyolo = {} 
        print('TIME',time.time()-t)


    def eigsorted(self, cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]



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
        #print(msg)
        self.pose_stamp = float(str(msg.header.stamp))/1e9
        #self.pose_time = datetime.fromtimestamp(self.pose_stamp)
        self.position = msg.pose.position

        quat = msg.pose.orientation
        q_b_w = np.array([quat.x, quat.y, quat.z, quat.w])
        self.euler_angles = conv.quaternion_to_euler_angles(q_b_w)
        self.psi = self.euler_angles[2]

    def ais_callback(self, msg):
        ais_id = msg.id
        aisx = msg.pose.position.x
        aisy = msg.pose.position.y
        self.ais_detections[ais_id] = [aisx, aisy]
        #print(ais_id,aisx,aisy)

    def radar_callback(self, msg):
        #print('RADAR',msg)
        radar_track_id = msg.track_id
        posterior_pos = msg.posterior.pos_est
        posterior_vel = msg.posterior.vel_est
        self.posterior_pos_cov[radar_track_id] =    [[msg.posterior.pos_cov.var_x, msg.posterior.pos_cov.cor_xy],
                                                    [msg.posterior.pos_cov.cor_xy, msg.posterior.pos_cov.var_y]]
        #self.posterior_vel_cov[radar_track_id] =    [[msg.posterior.vel_cov.var_x, msg.posterior.vel_cov.cor_xy],
        #                                            [msg.posterior.vel_cov.cor_xy, msg.posterior.vel_cov.var_y]]

        x = posterior_pos.x
        y = posterior_pos.y
        xdot = posterior_vel.x
        ydot = posterior_vel.y

        self.radar_detections[radar_track_id] = np.array([[x], [y], [xdot], [ydot]]) 

        self.target[radar_track_id] = 0

        

    def detector(self, i, n, net, meta, image, thresh=0.3):
        h = self.height
        w = self.width
        corners = None
        #print('i 2 ',i)
        detect = dn.detect(net, meta, image, thresh)
        #print(detect)
        for d in detect:
            #print(d)
            #print(d[0])
            #print(d[1])
            #print(d[2])
            if d[0] == b'boat':
                #corners = (yolo_boxes_to_corners(d))

                #box = []
                #box2 = np.zeros(4)
                #for b in bbox[2]:
                #    box.append(b)
                xc = d[2][0]
                yc = d[2][1]
                xw  = d[2][2]
                yh  = d[2][3]
                xmin0 = xc - xw/2
                ymin0 = yc - yh/2
                xmax0 = xc + xw/2
                ymax0 = yc + yh/2

                #bbox[2] = box2
                #return bbox


                #print (d)
                #print(xmin, ymin, xmax, ymax, w, h)# = corners[2]
                if i < n:
                    xmin = float(xmin0) + (w//n)*i
                    ymin = float(ymin0) + (h//2)-(h//n)
                    xmax = float(xmax0) + (w//n)*i
                    ymax = float(ymax0) + (h//2)-(h//n)
                    #print(xmin, ymin, xmax, ymax)
                else:
                    xmin = float(xmin0)
                    ymin = float(ymin0)
                    xmax = float(xmax0)
                    ymax = float(ymax0)
                #lst = list(d)
                #print(lst)
                #corners[2] = ((xc+((w//3)*i)),(yc+((h//3))),w1,h1)
                #print(lst)
                #objects.append(lst)

            
                if corners is None:
                    corners = []
                if ymin < self.height/2+20 and ymax > self.height/2-20:     #Only around the horizon
                    #pose = self.warppose[2] - self.psi
                    #dx = self.angle_2_pixel(pose)
                    #print('DX:   ',dx)
                    corners.append([d[0],d[1],np.array([xmin,ymin,xmax,ymax])])
                    if  abs(xw) > 4 and abs(yh) > 4:
                        cv2.rectangle(self.draw2, (int(xmin), int(ymin)), (int(xmax), int(ymax)), [0,0,255], 2)
                        bb_angle = self.fov_pixel_hor*(xmin + (xmax-xmin)/2 - self.width/2)
                        self.bb_angles.append(bb_angle)# + self.looking_angle + self.camera_offset - self.psi)
                        print(bb_angle)
                        camera_pixel = int((bb_angle)/self.fov_pixel_hor)
                        self.c_detections.append(camera_pixel)
                        print('detected')
        self.corners = corners  
        self.detect_ready = True 
    

    def feedback_callback(self, msg):
        print('DARK FEEDBACK',msg)
        #result = self.dark_client.get_result()
    def darknet_callback(self, status, result):
        #print('DARKNET feedback:  ', status, result.id)

        corners = None
        h = self.height
        w = self.width
        hwindow, wwindow = self.window.shape[:2]
        n = self.total_images

        if status == 3:
            self.dark_stamp = float(str(result.bounding_boxes.header.stamp))/1e9
            i = self.dark_id = int(result.id)
            try:
                angle = self.yolo_angle[i]
            except:
                angle = 0
            dark_boxes = result.bounding_boxes.bounding_boxes
            if dark_boxes != []:
                for d in dark_boxes:
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
                    '''
                    if i < n:
                        xmin = float(xmin)# + (w//n)*i
                        ymin = float(ymin)# + (h//2)-(h//n)
                        xmax = float(xmax)# + (w//n)*i
                        ymax = float(ymax)# + (h//2)-(h//n)
                    else:
                    '''
                    xmin = float(xmin)
                    ymin = float(ymin)
                    xmax = float(xmax)
                    ymax = float(ymax)
                    #self.bb_anglesyolo = []
                    if (obj == '"boat"' or obj == '"surfboard"')and prob > 0.15:
                        if corners is None:
                            corners = []
                        array = []
                        array.append(xmin)
                        array.append(ymin)
                        array.append(xmax)
                        array.append(ymax)
                        a = np.array([int(self.detection[i]-wwindow/2), ((h//2)-(h//n)/2), int(self.detection[i]-wwindow/2), ((h//2)-(h//n)/2)]) #-(w//n)/2+w/2
                        c = np.add(array, a)
                        corners.append([obj,prob,np.array([xmin,ymin,xmax,ymax])])
                        
                        bb_angle = (self.fov_pixel_hor*(c[0]+(c[2]-c[0])/2))  
                        camera_pixel = int((bb_angle)/self.fov_pixel_hor)
                        self.c_detections.append(camera_pixel)
                        self.bb_anglesyolo[i] = bb_angle + self.psi + float(angle)
                        
                self.corners = corners
                self.darkpose = self.euler_angles
            self.detect_ready = True
        

    def call_server(self,image, index):
        bridge = CvBridge()
        im = bridge.cv2_to_imgmsg(image, 'bgr8')#encoding="passthrough")
        goal = CheckForObjectsGoal()
        goal.id = index
        goal.image = im
        self.dark_client.send_goal(goal, done_cb = self.darknet_callback, feedback_cb=self.feedback_callback)



    def image_callback(self, msg, number):
        #print(number)
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w = img.shape[:2]
        '''
        [[2.33059692e+03 0.00000000e+00 1.05054296e+03]
         [0.00000000e+00 2.40544541e+03 1.20236899e+03]
         [0.00000000e+00 0.00000000e+00 1.00000000e+00]] 

         [[-0.61446217  0.54264669 -0.00584968 -0.01961701 -0.21955527]] 

        [[1.63663691e+03 0.00000000e+00 1.02370792e+03]
         [0.00000000e+00 1.64782940e+03 1.21251137e+03]
         [0.00000000e+00 0.00000000e+00 1.00000000e+00]] 

         [[-3.77040452e-01  1.63930763e-01  1.18141456e-04 -7.53546679e-04 -3.30016468e-02]] 

        [[1.35041667e+03 0.00000000e+00 1.03858079e+03]
         [0.00000000e+00 1.35274414e+03 1.21910677e+03]
         [0.00000000e+00 0.00000000e+00 1.00000000e+00]] 

         [[-2.93594198e-01  9.24909708e-02 -7.95066732e-04  1.54260738e-04 -1.29375283e-02]] 

        [[3.95727787e+03 0.00000000e+00 1.08515402e+03]
         [0.00000000e+00 6.68367696e+03 1.16446525e+03]
         [0.00000000e+00 0.00000000e+00 1.00000000e+00]] 

         [[-2.30366701e+00  1.62378114e+01  1.69321678e-02 -2.40693536e-02 -5.05074466e+01]] 

        [[1.63663691e+03 0.00000000e+00 1.02370792e+03]
         [0.00000000e+00 1.64782940e+03 1.21251137e+03]
         [0.00000000e+00 0.00000000e+00 1.00000000e+00]] 

         [[-3.77040452e-01  1.63930763e-01  1.18141456e-04 -7.53546679e-04 -3.30016468e-02]] 

        [[1.31005536e+03 0.00000000e+00 1.02682398e+03]
         [0.00000000e+00 1.30521483e+03 1.29066719e+03]
         [0.00000000e+00 0.00000000e+00 1.00000000e+00]] 

        [[-0.28836847  0.09049513 -0.00382027 -0.00036971 -0.01279517]] 
        '''
        if self.newcameramtx is None:
            #self.focal = 1350
            #mtx = np.matrix('1350.41716 0.0 1038.58110; 0.0 1352.74467 1219.10680; 0.0 0.0 1.0')
            #self.mtx = np.matrix('1350.0 0.0 1024.0; 0.0 1350.0 1232.0; 0.0 0.0 1.0')
            self.mtx = np.matrix('1600.0 0.0 1024.0; 0.0 1600.0 1232.0; 0.0 0.0 1.0')
            #distort = np.array([-0.293594324, 0.0924910801, -0.000795067830, 0.000154218667, -0.0129375553])
            #self.distort = np.array([-0.29, 0.09, -0.0, 0.0, -0.013])
            #self.distort = np.array([-0.38, 0.16, -0.0001, -0.00075, -0.033])
            self.distort = np.array([-0.38, 0.16, 0.0, 0.0, -0.033])

            self.newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.mtx,self.distort,(w,h),1,(w,h))

        # crop the image
        cropx, cropy = [216, 600]

        dst = cv2.undistort(img, self.mtx, self.distort, None, self.newcameramtx)
        h, w = dst.shape[:2]
        self.number = number
        #self.looking_angle = self.Mounting_angle*self.number
        self.image[number] = dst[cropy:h-cropy, cropx:w-cropx]
        self.im_attitude[number] = self.euler_angles
        #self.image = dst[cropy:h-cropy, cropx:w-cropx]
        #self.im_attitude = self.euler_angles
        
        self.height, self.width = self.image[number].shape[:2]

        self.newimage[number] = True
        #print(self.height, self.width)

    def bb_callback(self, msg, number):
        h = self.height
        w = self.width
        hwindow, wwindow = self.window.shape[:2]
        n = self.total_images
        angle = self.yolo_angle[number]
        #print(msg)
        if msg is not None:
            array = msg.data
            if array == []:
                print('Empty bounding box')
            else:
                xmin = array[0]
                ymin = array[1]
                xmax = array[2]
                ymax = array[3]
                
                a = np.array([int(self.detection[number]-wwindow/2), ((h//2)-(h//n)/2), int(self.detection[number]-wwindow/2), ((h//2)-(h//n)/2)]) #-(w//n)/2+w/2
                c = np.add(array, a)
                #print('ABC',array,c)
                if ((abs(xmax-xmin) > 4) and (abs(ymax-ymin) > 4)):
                    cv2.rectangle(self.draw2, (int(xmin), int(ymin)), (int(xmax), int(ymax)), [0,0,255], 2)
                    cv2.rectangle(self.draw, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), [0,0,255], 2)
                    bb_angle = (self.fov_pixel_hor*(c[0]+(c[2]-c[0])/2))# + self.looking_angle  
                    camera_pixel = int((bb_angle)/self.fov_pixel_hor)

                    self.c_detections.append(camera_pixel)
                    self.bb_angles = []
                    self.bb_angles.append(bb_angle + self.psi + float(angle)) #+ self.camera_offset)
            #print('re3_track')
            
        
    def re3_track(self, image, angle, corners=None):  
        h = self.height
        w = self.width
        hwindow, wwindow = self.window.shape[:2]
        n = self.total_images
        global initialize, boxToDraw#,tracker
        if corners is not None:
            if boxToDraw is None:# and all(corners) > 0:
                boxToDraw = corners
            #iou = bb_intersection_over_union(boxToDraw,corners)
            #if iou < 0.3:
            if corners[0] < 0 or corners[1] < 0 or corners[2] > wwindow or corners[3] > hwindow:
                print('Outside window')
            else:
                boxToDraw = tracker.track(image[:,:,::-1], 'Cam', corners)
                self.penalty = 0
        else:
            try:
                self.penalty += 1
                if self.penalty < 10:
                    boxToDraw = tracker.track(image[:,:,::-1], 'Cam')
                else:
                    boxToDraw = [0,0,0,0]
            except:
                print("No Bbox to track")

        if boxToDraw is not None:
            if any(boxToDraw) != 0:
                #bb_angles = []
                #for a in boxToDraw:
                a = np.array([((h//2)-(h//n)/2), int(angle-(w//n)/2+w/2), ((h//2)-(h//n)/2), int(angle-(w//n)/2+w/2)])
                b = np.array(boxToDraw)
                c = np.add(b, a)
                print('ABC',a,b,c)
                #self.window = self.warp[int((h//2)-(h//n)/2):int((h//2)+(h//n)/2),int(det-(w//n)/2+w/2):int(det+(w//n)/2+w/2)]
                #print(a,b)
                if ((abs(b[0]-b[2]) > 4) and (abs(b[1]-b[3]) > 4)):
                    cv2.rectangle(self.draw2, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), [0,0,255], 2)
                    cv2.rectangle(self.draw, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), [0,0,255], 2)
                    #self.bb_angles.append(self.fov_pixel*(b[0]+(b[2]-b[0])/2 - self.width/2)+self.looking_angle + self.camera_offset + self.psi) 
                    #self.bb_angles.append(self.fov_pixel*(xmin+(xmax-xmin)/2-self.width/2)+self.looking_angle+psi)
                    #print(bb_angle)
                    #self.bb_angles.append(bb_angle)
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
                    #print('BBBBBBBBBBBBB',bvalue)
                    if ((abs(bvalue[0]-bvalue[2]) > 4) and (abs(bvalue[1]-bvalue[3]) > 4)):
                        iou = bb_intersection_over_union(bvalue,cvalue)
                        if iou == 0:
                            #print(iou)
                            try:
                                new = True
                                self.penalty[bkey] += 1
                                #print('PENALTY',bkey,self.penalty)
                                if self.penalty[bkey] > 10:        # If no detections for X iterations. Remove track
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
                    if self.penalty[bkey] > 10:
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
                #bb_angles = []
                for a in boxToDraw:
                    b = boxToDraw[a]
                    #print(a,b)
                    if ((abs(b[0]-b[2]) > 4) and (abs(b[1]-b[3]) > 4)):
                        cv2.rectangle(self.draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), 
                            [0,0,255], 2)
                        self.bb_angles.append(self.fov_pixel_hor*(b[0]+(b[2]-b[0])/2 -self.width/2)+self.looking_angle + self.psi)
                        #print(bb_angle)
                        #self.bb_angles.append(bb_angle)
                #self.new_camera = True
                

                    
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
        x_angle = x*self.fov_pixel_hor-(self.width/2)*self.fov_pixel_hor+self.Mounting_angle
        y_angle = y*self.fov_pixel_hor-(self.height/2)*self.fov_pixel_hor
        z_angle = z
        return [x_angle, y_angle, z_angle] 

    def angle_2_pixel(self, r):
        return (r/self.fov_pixel_hor)#+(self.width/2)#-np.deg2rad(self.Mounting_angle*self.fov_pixel)

        
    def rotate_along_axis(self, image, number, phi=0, theta=0, psi=0, dx=0, dy=0, dz=0):
        # Get ideal focal length on z axis
        dz = self.focal*1.

        # Get projection matrix
        self.mat = self.get_M(phi, theta, psi, dx, dy, dz)
        #self.warp[number] = 
        return cv2.warpPerspective(image, self.mat, (self.width, self.height))

        

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

    #def loop(self, cam, pix):


    def start(self, number=0):
        #self.EKF_init()
        self.cameranumber = int(number)
        self.rate = rospy.Rate(self.updateRate)
        n = self.total_images = 6   #Antall bilder kuttet langs horisonten
        self.window = None
        h = self.height = 1264
        w = self.width = 1616
        self.looking_angle = 0 #np.deg2rad(self.Mounting_angle*self.number)
        if self.looking_angle > np.pi:
            self.looking_angle -= 2*np.pi
        elif self.looking_angle < -np.pi:
            self.looking_angle += 2*np.pi
        
        # Subscribers
        rospy.Subscriber('/seapath/pose',geomsg.PoseStamped, self.pose_callback)
        rospy.Subscriber('/ais/syn_text',vismsg.Marker, self.ais_callback)
        rospy.Subscriber('/ais/marker_syntext',vismsg.Marker, self.ais_callback)
        rospy.Subscriber('/radar/estimates', automsg.RadarEstimate, self.radar_callback)
        rospy.Subscriber('/ladybug/camera0/image_raw', Image, self.image_callback, 0)
        rospy.Subscriber('/ladybug/camera1/image_raw', Image, self.image_callback, 1)
        rospy.Subscriber('/ladybug/camera2/image_raw', Image, self.image_callback, 2)
        rospy.Subscriber('/ladybug/camera3/image_raw', Image, self.image_callback, 3)
        rospy.Subscriber('/ladybug/camera4/image_raw', Image, self.image_callback, 4)
        rospy.Subscriber('/re3/bbox0', stdmsg.Float32MultiArray , self.bb_callback, 0)
        rospy.Subscriber('/re3/bbox1', stdmsg.Float32MultiArray , self.bb_callback, 1)
        rospy.Subscriber('/re3/bbox2', stdmsg.Float32MultiArray , self.bb_callback, 2)
        rospy.Subscriber('/re3/bbox3', stdmsg.Float32MultiArray , self.bb_callback, 3)
        rospy.Subscriber('/re3/bbox4', stdmsg.Float32MultiArray , self.bb_callback, 4)
        # Publishers
        self.dark_client = actionlib.SimpleActionClient('darknet_ros/check_for_objects', darknetmsg.CheckForObjectsAction)
        bb_publisher0 = rospy.Publisher('/re3/bbox_new0', stdmsg.Float32MultiArray, queue_size=1)
        bb_publisher1 = rospy.Publisher('/re3/bbox_new1', stdmsg.Float32MultiArray, queue_size=1)
        bb_publisher2 = rospy.Publisher('/re3/bbox_new2', stdmsg.Float32MultiArray, queue_size=1)
        bb_publisher3 = rospy.Publisher('/re3/bbox_new3', stdmsg.Float32MultiArray, queue_size=1)
        bb_publisher4 = rospy.Publisher('/re3/bbox_new4', stdmsg.Float32MultiArray, queue_size=1)
        im_publisher0 = rospy.Publisher('/re3/image0', Image, queue_size=1)
        im_publisher1 = rospy.Publisher('/re3/image1', Image, queue_size=1)
        im_publisher2 = rospy.Publisher('/re3/image2', Image, queue_size=1)
        im_publisher3 = rospy.Publisher('/re3/image3', Image, queue_size=1)
        im_publisher4 = rospy.Publisher('/re3/image4', Image, queue_size=1)
        cam_publiser = rospy.Publisher('/re3/camera', stdmsg.Int8, queue_size=1)
        #track_publisher = rospy.Publisher('/re3/number', stdmsg.Int32, queue_size=1)

        print('FOV',self.fov_pixel_hor,self.fov_radians_hor)

        while not rospy.is_shutdown():
            t = time.time()
            corner = []
            c = []
            i = 0

                  
            for i in self.detections:
                detect = self.detections[i]
                det = detect + self.radar_offset
                #print(det)
                if det > np.pi:
                    det -= 2*np.pi
                elif det < -np.pi:
                    det += 2*np.pi
                #pix = det/self.fov_pixel_hor
                #print(det)
                cam = None
                #try:
                if det > (0*self.Mounting_angle - np.deg2rad(40)) and det < (0*self.Mounting_angle + np.deg2rad(40)):
                    #print('0')
                    pix = (det+0*self.Mounting_angle)/self.fov_pixel_hor
                    cam = 0
                elif det > (1*self.Mounting_angle - np.deg2rad(40)) and det < (1*self.Mounting_angle + np.deg2rad(40)):
                    pix = (det-1*self.Mounting_angle)/self.fov_pixel_hor
                    cam = 1
                elif det > (2*self.Mounting_angle - np.deg2rad(40)) and det < np.deg2rad(180):
                    pix = (det-2*self.Mounting_angle)/self.fov_pixel_hor
                    cam = 2
                elif det < -(1*self.Mounting_angle - np.deg2rad(40)) and det > -(1*self.Mounting_angle + np.deg2rad(40)):
                    #print('4')
                    pix = (det+1*self.Mounting_angle)/self.fov_pixel_hor
                    cam = 4
                elif det < -(2*self.Mounting_angle - np.deg2rad(40)) and det > np.deg2rad(-180):
                    #print('3')
                    pix = (det+2*self.Mounting_angle)/self.fov_pixel_hor
                    cam = 3
                else:
                    print('None of the above')
                #print(cam)
                if cam is not None:

                    cam_publiser.publish(cam)
                    self.looking_angle = self.Mounting_angle * cam
                    self.number = cam

                    if cam in self.newimage:

                        if self.newimage[cam] == True:
                            self.newimage[cam] = False
                            
                            try:
                                phi, theta, psi = self.im_attitude[cam]
                            except:
                                phi, theta, psi = self.euler_angles
                            
                            #if self.image[cam] is not None:
                            #    self.rotate_along_axis(self.image[cam], cam, -phi, -theta, -self.looking_angle,
                            if self.image[cam] is not None:
                                #print('hey')
                                warp = self.rotate_along_axis(self.image[cam], cam, -phi, -theta, -self.looking_angle,
                                    0,-(theta/self.fov_pixel_ver)*np.cos(self.looking_angle)+(phi/self.fov_pixel_ver)*np.sin(self.looking_angle),0)
                                #self.warppose = self.euler_angles
                                self.imcount+=1
                                self.window = warp[int((h//2)-(h//n)/2):int((h//2)+(h//n)/2),int(pix-(w//n)/2+w/2):int(pix+(w//n)/2+w/2)]
                                self.draw = warp.copy()
                                self.draw2 = self.window.copy()

                                bridge = CvBridge()
                                rosImg = bridge.cv2_to_imgmsg(self.window)#, encoding="passthrough")
                                #self.detector(i, n, net, meta, self.window, 0.1)
                                cam = 0
                                self.detection[cam] = pix
                                if (self.imcount % 2 == 0): 
                                    if self.detect_ready:
                                        #self.yolo_attitude[i] = self.im_attitude[cam]
                                        self.yolo_angle[cam]= self.looking_angle.copy()
                                        self.call_server(self.window, cam)
                                        self.detect_ready = False
                                    elif (self.count % 13 == 0):
                                        self.detect_ready = True

                                
                                locals()['im_publisher%s' %cam].publish(rosImg)
                                #('im_publisher%s' %cam).publish(rosImg)
                                
                                if self.corners is None:
                                    hh, ww = self.window.shape[:2]
                                    a0 = [ww/2,hh/2,ww/2,hh/2]
                                    b = stdmsg.Float32MultiArray(data=a0)
                                    #track_publisher.publish(int(i))
                                    #locals()['bb_publisher%s' %cam].publish(b)
                                else:
                                    self.corners.sort(reverse = True, key=lambda x :x[1])
                                    
                                    #if self.corners == []:
                                        #self.template_tracker(self.warp)
                                        #self.re3_track(self.window, det)
                                        #self.re3_multi_track(self.warp)
                                    #    im_publisher.publish(rosImg)
                                    if self.corners != []:#else:
                                        # phi_0, theta_0, psi_0 = self.yolo_attitude
                                        # delta_psi = psi - psi_0
                                        # delta_phi = phi - phi_0
                                        # delta_theta = theta - theta_0
                                        # y_change = -(delta_theta/self.fov_pixel_ver)*np.cos(self.looking_angle)+(delta_phi/self.fov_pixel_ver)*np.sin(self.looking_angle)
                                        # x_change = delta_psi/self.fov_pixel_hor
                                        # #print('DX, DY:   ',x_change, y_change)
                                        # a_change = [x_change,y_change,x_change,y_change]
                                        
                                        a0 = self.corners[0][2]
                                        #a = np.add(a0, a_change)
                                        #print(a0,a_change,a)
                                        b = stdmsg.Float32MultiArray(data=a0)
                                        #im_publisher.publish(rosImg)
                                        #track_publisher.publish(int(i))
                                        locals()['bb_publisher%s' %cam].publish(b)

                                    self.corners = None
                                #else:
                                    #self.template_tracker(self.warp)
                                    #self.re3_track(self.window,det)
                                    #self.re3_multi_track(self.warp)
                                #    im_publisher.publish(rosImg)
                                
                            cv2.line(self.draw, (pix+int(w/2), 0), (pix+int(w/2), h), (255,0,0), 10)
                                #cv2.imshow('Cam', self.draw2)
                break
                            #except:
                            #    print('Not valid detection angle')

            self.detections = {}

            

            if self.draw is not None:
                for c in self.c_detections:
                    cv2.line(self.draw, (c+int(w/2), 0), (c+int(w/2), h), (0,255,0), 8)
                self.c_detections = []
                cv2.imshow('Cam', self.draw)
                cv2.waitKey(1)

            #plt.xlabel('Iterations')
            #plt.ylabel('Distance[m]')
            plt.xlabel('Easting')
            plt.ylabel('Northing')
            #plt.title('About as simple as it gets, folks')
            #plt.grid(True)
            plt.legend(shadow=True, fancybox=True)
            #plt.axes().set_aspect('equal', 'datalim')
            #plt.plot(float(self.position.y), float(self.position.x), '+b')
            plt.draw()   
            plt.pause(0.0000001)

                    #self.previmg = self.window

            self.extended_kalman()
            self.count+=1
            print('LOOP',time.time()-t)
            self.rate.sleep()



# Main function
if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--number", "-n", help="set camera number")
    #args = parser.parse_args()
    #if args.number:
    #    number = args.number  
    #    print("set camera number to %s" % args.number)

    #cv2.namedWindow('Cam', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('Cam2', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Cam', cv2.WINDOW_NORMAL) 

    #net = dn.load_net(b"/home/runar/yolov3.cfg", b"/home/runar/yolov3.weights", 0)
    #meta = dn.load_meta(b"/home/runar/coco.data")
    
    #tracker = re3_tracker.Re3Tracker()
    rospy.init_node("CameraTracker")
    # Setup Telemetron ownship tracker
    #telemetron_tf = TransformListener()

    DetectObjects_node = DetectObjects()   
    #DetectObjects_node.start(number)
    DetectObjects_node.start()