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


np.set_printoptions(precision=10)
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
        jacVal = self.fJac(self.previousEstimate) #+ self.processNoise
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
                gain[:,[0,1]] = 0
            else:
                gain = self.gain
                gain[:,[2]] = 0
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
            #print(self.estimate)
            jacVal = self.hJac(self.estimate)
            invVal = np.dot(jacVal, np.dot(self.errorPrediction, np.transpose(jacVal))) + self.R
            self.gain = np.dot(self.errorPrediction, np.dot(np.transpose(jacVal) , np.linalg.inv(invVal) ))
            self.errorPrediction = np.dot((np.identity(self.numStates) - np.dot(self.gain, jacVal)), self.errorPrediction)
        
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
        self.previousErrorPrediction = self.errorPrediction;
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
        self.prev_dark_stamp = 0
        self.darkpose = [0,0,0]
        self.millisecond = None
        self.firstimage = None
        self.second = None
        self.minute = None
        self.hour = None
        self.newimagetimestamp = 0
        self.radar_pixel = 0
        self.detections = []
        self.c_detections = []
        self.radar_detections = {}
        self.angle_ned = 0
        self.posterior_pos_cov = {}
        self.bb_angles = []
        self.range = 100
        self.track_id = None
        self.looking_angle = 0
        self.focal = 1350
        self.number = 4
        self.Mounting_angle = 72       # 5 cameras, 360/5=72
        self.camera_offset = np.deg2rad(-3)        # psi degrees between camera and vessel
        self.radar_offset = np.deg2rad(3)         # psi degrees between radar and vessel
        self.fov_radians = np.deg2rad(100)      #FOV is about 100 deg
        self.fov_pixel = self.fov_radians/1616#self.width

        self.matrix = None
        self.penalty = np.zeros([100])

        # Estimation parameter of EKF
        self.updateRate = 50
        self.dT = 1./self.updateRate    #Timestep
        self.decay = 0.95
        sigma_cv = 0.4    #Process noise strength
        sigma_r = 0.1   #Sensor noise strength
        sigma_c = 0.1   #Sensor noise strength
        #self.processNoise = np.diag([1.0, 1.0, 1.0, 1.0])**2  # predict state covariance
        self.radarNoise = sigma_r**2 * np.identity(3) #diag([10., 10.])**2  # Observation x,y position covariance
        self.cameraNoise = sigma_c**2 * np.identity(3)
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
        print(self.processNoiseCovariance)
    
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
        initialCovariance = [[cov[0][0],cov[0][1],0,0],
                             [cov[1][0],cov[1][1],0,0],
                             [0,0,0,0],
                             [0,0,0,0]]
        self.kf[i] = ExtendedKalman(initialReading,initialCovariance, 1,4,self.sensorNoise, self.f, self.h, self.processNoiseCovariance)


    def extended_kalman(self):
        #print('EKF: ',msg)

        #phi, theta, psi = self.euler_angles
        y = None
        R = self.radarNoise
        angle_only = False
        #b = None
        radar_detections2 = self.radar_detections.copy()
        for i in radar_detections2:
            est_x = radar_detections2[i][0]-self.position.x
            est_y = radar_detections2[i][1]-self.position.y
            est_range = np.sqrt(est_x**2+est_y**2)
            if i not in self.kf and est_range < 1000:
                self.EKF_init(i, radar_detections2[i], self.posterior_pos_cov[i])
                self.kf[i].predict()
                self.myEstimate[i] = self.kf[i].getEstimate()
        for i in self.target:
            if self.target[i] > 100:
                del self.kf[i]
                self.target[i] = -1

        for i in self.kf:
            self.kf[i].predict()

            est_x = self.myEstimate[i][0]-self.position.x
            est_y = self.myEstimate[i][1]-self.position.y
            est_xdot = self.myEstimate[i][2]
            est_ydot = self.myEstimate[i][3]
            if abs(est_xdot) > 100 or abs(est_ydot) > 100:
                self.target[i] = 2000
            self.angle_ned = np.arctan2(est_y,est_x)
            angle_body = self.angle_ned - self.psi - self.looking_angle + self.radar_offset     # Installation angle offset between radar and vessel  
            
            #print('AAAAA',angle_body,self.bb_angles)
            
            if angle_body < -np.pi:
                angle_body += 2*np.pi
            elif angle_body > np.pi:
                angle_body -= 2*np.pi  
            
            #print('xy',est_x,est_y)
            est_range = np.sqrt(est_x**2+est_y**2)
            #print(psi, angle_ned, angle_body, est_range)
            
            for j in radar_detections2:
                if i == j and est_range < 1000:
                    x0 = radar_detections2[j][0]
                    x1 = radar_detections2[j][1]
                    radar_angle = np.arctan2(x1-self.position.y, x0-self.position.x)
                    #radar_range = np.sqrt((x1-self.position.y)**2 + (x0-self.position.x)**2)
                    y = np.array([x0,x1,radar_angle]).reshape(-1,1)
                    R = self.radarNoise
                    plt.plot(float(x1), float(x0), '+r')
                    g = self.kf[i].getGain()
                    p = self.kf[i].getP()
                    est = self.kf[i].getEstimate()
                    #print('Radar:   ',i, angle_body)
                    #print('est:  ',y,est)
                    #print(g)
                    #print(p)
                    
                    break
            else:
                if est_range < 500:
                    a = []
                    for k in self.bb_angles:
                        #camera_pixel = int(k/self.fov_pixel)
                        #self.c_detections.append(camera_pixel)
                        print('ANGLE: ',abs(self.angle_ned-k),k,self.angle_ned, self.psi)
                        a.append([abs(self.angle_ned-k), k])
                    if a != []:
                        #print(a)
                        b = min(a, key=lambda x: x[0])
                        c = b[1] - self.looking_angle + self.camera_offset - self.psi
                        #print(np.rad2deg(c),c, b, max(a, key=lambda x: x[0]))
                        y = np.array([0,0, c]).reshape(-1,1)
                        R = self.cameraNoise 
                        angle_only = True
                        
                        #if abs(angle_body - k) < np.deg2rad(10):
                        #a.append([abs(angle_body - k), k])
                        #print('bbb',b)
                        #if a != []:
                        #b = min(abs(a[0]))
                        #b = min(a, key=lambda x: x[0])
                        dx = float(est_range*np.cos(b[1]))
                        dy = float(est_range*np.sin(b[1]))
                        #if abs(est_x-dx) < 100 and abs(est_y-dy) < 100 and b[0] < np.deg2rad(10):
                        #print(dx,dy)
                        #if b[0] < np.deg2rad(5):
                            #y = np.array([[dx+self.position.x], [dy+self.position.y]])  
                            #R = self.cameraNoise 
                        #camera_angle_image = k #- self.camera_offset #- self.looking_angle
                        #camera_pixel = int(b[1]/self.fov_pixel)
                        #print('PIXEL',camera_pixel)
                        #bb_angle = self.fov_pixel*(xmin + (xmax-xmin)/2 + self.width/2)
                        #camera_pixel = int(bb_angle/self.fov_pixel)
                        #self.c_detections.append(camera_pixel)
                            #print('Camera  ',i)
                            #g = self.kf[i].getGain()
                            #p = self.kf[i].getP()
                            #est = self.kf[i].getEstimate()
                        plt.plot(float(dy+self.position.y), float(dx+self.position.x), 'ob')
                        #print('est:  ',y)
                        #print(g)
                        #print(p)
                        
            if est_range < 1000:
                radar_angle_image = angle_body #-np.pi/4
                #print(radar_angle_image, angle_body, self.looking_angle)
                radar_pixel = int(radar_angle_image/self.fov_pixel)
                self.detections.append(radar_pixel)
                
            #if y is not None:
            self.kf[i].update(y, R, angle_only)
            #grab result for this iteration 
            self.myEstimate[i] = self.kf[i].getEstimate()
            '''
            est_x = self.myEstimate[i][0]-self.position.x
            est_y = self.myEstimate[i][1]-self.position.y
            est_xdot = self.myEstimate[i][2]
            est_ydot = self.myEstimate[i][3]
            if abs(est_xdot) > 100 or abs(est_ydot) > 100:
                self.target[i] = 2000
            angle_ned = np.arctan2(est_y,est_x)
            angle_body = angle_ned - self.psi - self.looking_angle + np.deg2rad(self.radar_offset)     # Installation angle offset between radar and vessel  

            if angle_body < -np.pi:
                angle_body += 2*np.pi
            elif angle_body > np.pi:
                angle_body -= 2*np.pi  
            
            #print('xy',est_x,est_y)
            est_range = np.sqrt(est_x**2+est_y**2)
            #print(i,len(self.kf), self.myEstimate[i])
            self.target[i] += 1
            '''
            plt.plot(float(self.myEstimate[i][1]), float(self.myEstimate[i][0]), '+k')

        self.radar_detections = {}
        self.bb_angles = []  
        #self.bb_angles.append(-self.angle_ned)

        



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
        self.psi = self.euler_angles[2]
        #print('ANGLES', self.euler_angles)
        #print(self.mtx, self.dist, self.rvec, self.tvec)

    def radar_callback(self, msg):
        #print('RADAR',msg)
        #radar_stamp = float(str(msg.header.stamp))/1e9
        #print(self.radar_stamp, self.pose_stamp)
        radar_track_id = msg.track_id
        posterior_pos = msg.posterior.pos_est
        posterior_vel = msg.posterior.vel_est
        self.posterior_pos_cov[radar_track_id] =    [[msg.posterior.pos_cov.var_x, msg.posterior.pos_cov.cor_xy],
                                                    [msg.posterior.pos_cov.cor_xy, msg.posterior.pos_cov.var_y]]
        #print(self.posterior_pos_cov)
        #posterior_vel_cov = msg.posterior.vel_cov
        #print(self.posterior_pos, self.posterior_vel)
        #self.posterior_ned = np.array([posterior_pos.x, posterior_pos.y])
        #print(self.radar_stamp)
        #print(self.centroid_ned, self.ned)
        x = posterior_pos.x
        y = posterior_pos.y
        xdot = posterior_vel.x
        ydot = posterior_vel.y
        #dx = posterior_pos.x - self.position.x
        #dy = posterior_pos.y - self.position.y
        #phi, theta, psi = self.euler_angles

        #radar_angle_ned = np.arctan2(dx,dy)
        #radar_angle_body = radar_angle_ned - psi + np.deg2rad(-3.14)     # Installation angle offset between camera and radar  

        #if radar_angle_body < -np.pi:
        #    radar_angle_body += 2*np.pi
        #elif radar_angle_body > np.pi:
        #    radar_angle_body -= 2*np.pi

        #radar_angle_image = radar_angle_body - self.number*np.deg2rad(self.Mounting_angle)
        #radar_pixel = int(radar_angle_image/self.fov_pixel)
        #radar_range = np.sqrt(dx**2+dy**2)
        self.radar_detections[radar_track_id] = np.array([[x], [y], [xdot], [ydot]]) 
        #self.radar_detections[self.radar_track_id] = (np.array([self.radar_angle_body, self.radar_range, self.radar_track_id,
        #    [self.posterior_vel.x, self.posterior_vel.y], [self.posterior_pos_cov.var_x, self.posterior_pos_cov.var_y, self.posterior_pos_cov.cor_xy], 
        #    [self.posterior_vel_cov.var_x, self.posterior_vel_cov.var_y, self.posterior_vel_cov.cor_xy]]))
        #self.detections.append(radar_pixel)

        self.target[radar_track_id] = 0
        #cv2.line(self.warp, (self.radar_pixel, 0), (self.radar_pixel, self.height), (0,255,0), 10)
        #print('RADAR_ANGLE',radar_angle_ned, psi, self.radar_angle_body, self.radar_angle_image, self.radar_pixel)
        

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
                        bb_angle = self.fov_pixel*(xmin + (xmax-xmin)/2 - self.width/2)
                        self.bb_angles.append(bb_angle)# + self.looking_angle + self.camera_offset - self.psi)
                        print(bb_angle)
                        camera_pixel = int(bb_angle/self.fov_pixel)
                        self.c_detections.append(camera_pixel)
                        print('detected')
        self.corners = corners  
        self.detect_ready = True 
    

    def feedback_callback(self, msg):
        print('DARK FEEDBACK',msg)
        #result = self.dark_client.get_result()
    def darknet_callback(self, status, result):
        #print('DARKNET feedback:  ', status, result.id)
        #phi, theta, psi = self.euler_angles
        corners = None
        #corner = []
        h, w = self.window.shape[:2]
        #h = self.height
        #w = self.width
        n = self.total_images

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

                    if i < n:
                        xmin = float(xmin)# + (w//n)*i
                        ymin = float(ymin)# + (h//2)-(h//n)
                        xmax = float(xmax)# + (w//n)*i
                        ymax = float(ymax)# + (h//2)-(h//n)
                    else:
                        xmin = float(xmin)
                        ymin = float(ymin)
                        xmax = float(xmax)
                        ymax = float(ymax)

                    if obj == '"boat"' and prob > 0.1:
                        if corners is None:
                            corners = []
                        #if ymin < self.height/2+10 and ymax > self.height/2-10:     #Only around the horizon
                        pose = self.warppose[2] - self.psi
                        dx = 0#self.angle_2_pixel(pose)
                        #print('DX:   ',dx)
                        corners.append([obj,prob,np.array([xmin-10,ymin-10,xmax+10,ymax+10])])
                        
                        #a = {}
                        #bb_angle = (self.fov_pixel*(xmin+(xmax-xmin)/2 + dx) + self.looking_angle + self.detection)
                        #bb_angle = self.fov_pixel*(xmin+(xmax-xmin)/2 + dx - self.width/2)
                        #self.bb_angles.append(bb_angle)# + self.looking_angle + self.camera_offset + self.psi)
                        '''
                        for b in self.bb_angles:
                            a[b]=[abs(self.bb_angles[b][1]-bb_angle), self.bb_angles[b][1], self.bb_angles[b][2]]
                        if a != {}:
                            print(a)
                            c = min(a, key=lambda x: x[0])
                            if c[0] < np.deg2rad(2):
                                self.bb_angles[c][1]=(bb_angle + self.looking_angle + self.psi + np.deg2rad(self.camera_offset))#self.fov_pixel*(xmin+(xmax-xmin)/2 + dx - self.width/2)+ self.looking_angle + self.psi + np.deg2rad(self.camera_offset))
                            #print(self.bb_angles)
                            cv2.rectangle(self.draw, (int(xmin+dx), int(ymin)), (int(xmax+dx), int(ymax   )), [0,0,255], 2)
                            self.templates[bb_angle] = self.warp[ymin:(ymax-ymin),xmin:(xmax-xmin)]
                        '''
                        #camera_pixel = int(bb_angle/self.fov_pixel)
                        #self.c_detections.append(camera_pixel)

                self.corners = corners
                self.darkpose = self.euler_angles
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
        #print(self.height, self.width)

    def bb_callback(self, msg):
        h = self.height
        w = self.width
        hwindow, wwindow = self.window.shape[:2]
        n = self.total_images
        #print(msg)
        if msg is not None:
            array = msg.data
            xmin = array[0]
            ymin = array[1]
            xmax = array[2]
            ymax = array[3]
            
            a = np.array([int(self.detection-wwindow/2), ((h//2)-(h//n)/2), int(self.detection-wwindow/2), ((h//2)-(h//n)/2)]) #-(w//n)/2+w/2
            c = np.add(array, a)
            print('ABC',array,c)
            if ((abs(xmax-xmin) > 4) and (abs(ymax-ymin) > 4)):
                cv2.rectangle(self.draw2, (int(xmin), int(ymin)), (int(xmax), int(ymax)), [0,0,255], 2)
                cv2.rectangle(self.draw, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), [0,0,255], 2)
                bb_angle = (self.fov_pixel*(c[0]+(c[2]-c[0])/2))# + self.looking_angle 
                #bb_angle = (self.fov_pixel*(xmin+(xmax-xmin)/2 + dx) + self.looking_angle + self.detection)
                print(bb_angle, np.rad2deg(bb_angle))
                camera_pixel = int(bb_angle/self.fov_pixel)
                #angle_body = self.angle_ned - self.psi - self.looking_angle + self.radar_offset
                self.c_detections.append(camera_pixel)
                self.bb_angles = []
                self.bb_angles.append(bb_angle + self.psi + self.looking_angle + self.camera_offset)
                print(self.bb_angles,np.rad2deg(self.bb_angles[0]))
        
        
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
                        self.bb_angles.append(self.fov_pixel*(b[0]+(b[2]-b[0])/2 -self.width/2)+self.looking_angle + self.psi)
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
        x_angle = x*self.fov_pixel-(self.width/2)*self.fov_pixel+np.deg2rad(self.Mounting_angle)
        y_angle = y*self.fov_pixel-(self.height/2)*self.fov_pixel
        z_angle = z
        return [x_angle, y_angle, z_angle] 

    def angle_2_pixel(self, r):
        return (r/self.fov_pixel)#+(self.width/2)#-np.deg2rad(self.Mounting_angle*self.fov_pixel)
        #y = (ry/self.fov_pixel)#+(self.height/2)
        #z = 1
        #return [x, y, z]
        
    def rotate_along_axis(self, image, phi=0, theta=0, psi=0, dx=0, dy=0, dz=0):
        # Get ideal focal length on z axis
        dz = self.focal*1.
        #axis = np.float32([[3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)

        # Get projection matrix
        self.mat = self.get_M(phi, theta, psi, dx, dy, dz)
        #if len(np.shape(image)) > 2: 
        self.warp = cv2.warpPerspective(image, self.mat, (self.width, self.height))
        self.draw = self.warp.copy()
        #else:
        #    return cv2.warpPerspective(image,self.mat)
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
        #self.EKF_init()
        self.rate = rospy.Rate(self.updateRate)
        n = self.total_images = 6   #Antall bilder kuttet langs horisonten
        self.window = None
        h = self.height = 1264
        w = self.width = 1616
        self.looking_angle = np.deg2rad(self.Mounting_angle*self.number)
        if self.looking_angle > np.pi:
            self.looking_angle -= 2*np.pi
        elif self.looking_angle < -np.pi:
            self.looking_angle += 2*np.pi
        
        # Subscribers
        rospy.Subscriber('/seapath/pose',geomsg.PoseStamped, self.pose_callback)
        rospy.Subscriber('/radar/estimates', automsg.RadarEstimate, self.radar_callback)
        rospy.Subscriber('/ladybug/camera0/image_raw', Image, self.image_callback)
        rospy.Subscriber('/re3/bbox', stdmsg.Float32MultiArray , self.bb_callback)
        # Publishers
        self.dark_client = actionlib.SimpleActionClient('darknet_ros/check_for_objects', darknetmsg.CheckForObjectsAction)
        bb_publisher = rospy.Publisher('/re3/bbox_new', stdmsg.Float32MultiArray, queue_size=1)
        im_publisher = rospy.Publisher('/re3/image', Image, queue_size=1)

        print('FOV',self.fov_pixel,self.fov_radians)

        while not rospy.is_shutdown():

            corner = []
            self.extended_kalman()
            c = []
            i = 0

            if self.newimage == True:
                self.newimage = False
                
                
                #print(self.looking_angle)
                phi, theta, psi = self.euler_angles
                self.rotate_along_axis(self.image, -phi, -theta, -self.looking_angle,
                    0,-self.angle_2_pixel(theta)*np.cos(self.looking_angle)+self.angle_2_pixel(phi)*np.sin(self.looking_angle),0)
                self.warppose = self.euler_angles
                self.count+=1
                '''
                if self.ang is None:
                    self.ang = self.angle_ned
                if self.bb_angles is not None:
                    for b in self.bb_angles:
                        a = (b - self.looking_angle -self.camera_offset - self.psi)/self.fov_pixel
                        c.append(np.array([abs(b-self.angle_ned), a]))
                        print('AAAAAAAAAAA',a,b)
                    if c != []:
                        print('CCCCCCCC',c)
                        self.ang = min(c, key=lambda x: x[0])[1]
                        print('SELFANG',self.ang)
                        #self.bb_angle = []
                        #self.bb_angle.append(self.ang)
                        #self.window = self.warp[(h//2)-(h//n):(h//2)+(h//n),int(self.ang-(w//n)/2+w/2):int(self.ang+(w//n)/2+w/2)]
                        #wh, ww = self.window.shape[:2]
                 

                self.count+=1
                    
                if self.detect_ready:# and (self.count % 2 == 0):# or float(str(rospy.Time.now()))/1e9-self.dark_stamp > 20:
                    #if self.dark_id != self.prev_dark_id or self.prev_dark_id is None:
                    #    self.prev_dark_id = self.dark_id
                    #    self.detect_ready = False
                    self.index +=1
                    if self.index > n-1:
                        self.index = 0
                
                    i = self.index
                    #print('i 1 ',i)
                    #self.yolo_image = self.warp.copy()

                    self.prev_dark_stamp = float(str(rospy.Time.now()))/1e9
                    self.warppose = self.euler_angles

                    tile = self.warp[(h//2)-(h//n):(h//2)+(h//n),(w//n)*i:(w//n)*i+(w//n)]
                    if self.window is None:
                        self.window = tile.copy()
                    #cv2.imshow('Cam', tile)
                    #cv2.waitKey(1)

                    #self.call_server(tile, i)
                    #self.detector(i, n, net, meta, tile, 0.1)
                
                
                if self.corners is not None:
                    self.corners.sort(reverse = True, key=lambda x :x[1])
                    print(self.corners)
                    
                    #pose = np.subtract(self.warppose, self.euler_angles) #darkpose)
                    #for c in self.corners:
                        #corner.append(np.array(c[2]))
                        #d = c[2]
                        #corner = (np.array([d[0]-(self.ang[1]-(w//n)/2+w/2), d[1]-(h//2)-(h//n), d[2]-(self.ang[1]-(w//n)/2+w/2), d[3]-(h//2)-(h//n)]))
                        #break
                        #if corner == []:
                            #print(c)
                            #if c[2][3] < self.height/2+50:
                                #corner = [np.array(c[2])]
                                #print(corner, corner[0][1])
                                #dx = self.angle_2_pixel(pose[2])
                                #corner[0] = [corner[0][0]+dx,corner[0][1],corner[0][2]+dx,corner[0][3]]
                        #else:
                        #    if c[2][3] < self.height/2+50:
                        #        corner.append(np.array(c[2]))
                    
                    if self.corners == []:
                        #self.template_tracker(self.warp)
                        self.re3_track(self.window)
                        #self.re3_multi_track(self.warp)
                    else:
                        #c = []
                        #c.append(corner[0])
                        #self.template_tracker(self.warp, corner[0])
                        d = self.corners[0][2]
                        print(d, self.ang)
                        corner = (np.array([d[0]-(self.ang-(w//n)/2+w/2), d[1]-(h//2)-(h//n), d[2]-(self.ang-(w//n)/2+w/2), d[3]-(h//2)-(h//n)]))
                        self.re3_track(self.window, corner)
                        #self.re3_multi_track(self.warp, corner)
                else:
                    #self.template_tracker(self.warp)
                    self.re3_track(self.window)
                    #self.re3_multi_track(self.warp)
                '''   
                for i in range(len(self.detections)):
                    self.detection = det = self.detections[i]
                    print(det)
                    self.window = self.warp[int((h//2)-(h//n)/2):int((h//2)+(h//n)/2),int(det-(w//n)/2+w/2):int(det+(w//n)/2+w/2)]
                    self.draw2 = self.window.copy()
                    #print(self.window.shape[:])#cv2.imshow('Cam', self.draw2)
                    if all(self.window.shape[:]) != 0:
                        bridge = CvBridge()
                        rosImg = bridge.cv2_to_imgmsg(self.window)#, encoding="passthrough")
                        #self.detector(i, n, net, meta, self.window, 0.1)
                        if (self.count % 5 == 0):# and self.detect_ready:
                            self.call_server(self.window, i)
                            self.detect_ready = False
                        
                        if self.corners is not None:
                            self.corners.sort(reverse = True, key=lambda x :x[1])
                            #print(self.corners)
                            #corner = (np.array([d[0]-(d-(w//n)/2+w/2), d[1]-(h//2)-(h//n), d[2]-(d-(w//n)/2+w/2), d[3]-(h//2)-(h//n)]))
                            if self.corners == []:
                                #self.template_tracker(self.warp)
                                #self.re3_track(self.window, det)
                                #self.re3_multi_track(self.warp)
                                im_publisher.publish(rosImg)
                            else:
                                #c = []
                                #c.append(corner[0])
                                #self.template_tracker(self.warp, corner[0])
                                #d = self.corners[0][2]
                                #print(d, self.detection)
                                #corner = (np.array([d[0]-(det-(w//n)/2+w/2), d[1]-(h//2)-(h//n), d[2]-(det-(w//n)/2+w/2), d[3]-(h//2)-(h//n)]))
                                #self.re3_track(self.window, det, d)
                                #self.re3_multi_track(self.warp, corner)
                                
                                a = self.corners[0][2]
                                b = stdmsg.Float32MultiArray(data=a)
                                im_publisher.publish(rosImg)
                                bb_publisher.publish(b)
                        else:
                            #self.template_tracker(self.warp)
                            #self.re3_track(self.window,det)
                            #self.re3_multi_track(self.warp)
                            im_publisher.publish(rosImg)

                        cv2.line(self.draw, (det+int(w/2), 0), (det+int(w/2), h), (255,0,0), 10)
                        cv2.imshow('Cam', self.draw2)
                        break
                
                self.detections = []
                for c in self.c_detections:
                    #print(d)
                    #self.window = self.warp[int((h//2)-(h//n)/2):int((h//2)+(h//n)/2),int(d-(w//n)/2+w/2):int(d+(w//n)/2+w/2)]
                    cv2.line(self.draw, (c+int(w/2), 0), (c+int(w/2), h), (0,255,0), 8)
                self.c_detections = []
                

                #cv2.line(self.draw, (d+int(w/2), 0), (d+int(w/2), h), (255,0,0), 10)
                cv2.imshow('Cam3', self.draw)
                cv2.waitKey(1)
                plt.plot(float(self.position.y), float(self.position.x), '+b')
                plt.draw()   
                plt.pause(0.001)
                self.newimage = False

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

    #net = dn.load_net(b"/home/runar/yolov3.cfg", b"/home/runar/yolov3.weights", 0)
    #meta = dn.load_meta(b"/home/runar/coco.data")
    
    #tracker = re3_tracker.Re3Tracker()
    rospy.init_node("CameraTracker")
    # Setup Telemetron ownship tracker
    #telemetron_tf = TransformListener()

    DetectObjects_node = DetectObjects()   
    DetectObjects_node.start()