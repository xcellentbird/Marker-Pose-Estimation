#!/usr/bin/env python
import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
import sys, time, math
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

#from geometry_msgs.msg import Accel
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Bool

myid = 72
marker_size = 4.115 #[cm]

#logitec cam parameter
cam_mat = np.matrix([[520.388401, 0.000000, 322.289271],
            [0.000000, 519.785609, 252.492932],
            [0.000000, 0.000000, 1.000000]])
cam_distort = np.matrix([0.025544, -0.182058, 0.007314, 0.005738, 0.000000])

font = cv2.FONT_HERSHEY_PLAIN

#--- 180 deg rotation matrix around the x axis
R_flip  = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] =  1.0
R_flip[1,1] = -1.0
R_flip[2,2] = -1.0

aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters  = aruco.DetectorParameters_create()

class ArUco:
    def __init__(self):
        self.frame = []
        self.freeze_cnt = 0
        self.freeze = False
        self.call_once_switch = True
        self.cant_find_mark = True
        self.gogo = False
        self.bridge = CvBridge()
        self.CAM_X = 0.0
        self.CAM_Z = 0.0
        self.CAM_PITCH = 0.0
        self.sub_image = rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.image_callback)
        self.sub = rospy.Subscriber("Go_Cam", Bool, self.cam_on_callback)
        self.pub = rospy.Publisher("Movement",Pose2D,latch=True,queue_size=1000)

    def image_callback(self, img_msg):
        #try:

        cv_image = self.bridge.compressed_imgmsg_to_cv2(img_msg, "passthrough")
        #except CvBridgeError, e:
            #rospy.logerr("CvBridge Error: {0}".format(e))
        self.frame = cv_image
        self.Image_Process()
        rospy.loginfo("x= %4.0f, z= %4.0f, pitch= %4.0f",self.CAM_X,self.CAM_Z,self.CAM_PITCH)
        cv2.imshow("Image Window", self.frame)
        cv2.waitKey(1)

    def cam_on_callback(self, data):
        p = Pose2D()
        p.x = abs(self.CAM_X)
        p.y = self.CAM_Z
        p.theta = self.CAM_PITCH

        freeze_cnt_max = 10
        if data.data == True :
            self.gogo = True
            self.freeze_cnt += 1
            if self.freeze_cnt > freeze_cnt_max and self.call_once_switch and not self.cant_find_mark:
                #self.freeze = True
                self.pub.publish(p)
                rospy.loginfo("I send x= %4.0f, z= %4.0f, pitch= %4.0f",abs(self.CAM_X),self.CAM_Z,self.CAM_PITCH)
                self.call_once_switch = False
            elif self.freeze_cnt <= freeze_cnt_max and self.call_once_switch and not self.cant_find_mark:
                rospy.loginfo("publish countdown = %d/%d ",self.freeze_cnt, freeze_cnt_max)
            elif self.freeze_cnt > freeze_cnt_max and self.call_once_switch and self.cant_find_mark:
                rospy.loginfo("there is no marker. please show me one")
        elif data.data == False:
            self.gogo = False
            self.freeze_cnt = 0
            self.call_once_switch = True
            #self.freeze = False

        #on_while = True
        #while on_while:
            #connections = self.pub.get_num_connections()
            #if connections > 0:
                #self.pub.publish(p)
                #rospy.loginfo("send Pose2D")
                #on_while = False

    def Image_Process(self):
        ret = self.frame
        gray    = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters,
                              cameraMatrix=cam_mat, distCoeff=cam_distort)
        if ids is not None : #and self.freeze is not True:
            self.cant_find_mark = False
            ret = aruco.estimatePoseSingleMarkers(corners, marker_size, cam_mat, cam_distort)
            rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]
            aruco.drawDetectedMarkers(self.frame, corners)
            aruco.drawAxis(self.frame, cam_mat, cam_distort, rvec, tvec, 10)

            str_position = "MARKER Position x=%4.0f  y=%4.0f  z=%4.0f"%(tvec[0], tvec[1], tvec[2])
            cv2.putText(self.frame, str_position, (0, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            R_ct    = np.matrix(cv2.Rodrigues(rvec)[0])
            R_tc    = R_ct.T

            roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip*R_tc)
            str_attitude = "MARKER Attitude r=%4.0f  p=%4.0f  y=%4.0f"%(math.degrees(roll_marker),math.degrees(pitch_marker),math.degrees(yaw_marker))
            cv2.putText(self.frame, str_attitude, (0, 150), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            pos_camera = -R_tc*np.matrix(tvec).T
            
            str_position = "CAMERA Position x=%4.0f  y=%4.0f  z=%4.0f"%(pos_camera[0], pos_camera[1], pos_camera[2])
            cv2.putText(self.frame, str_position, (0, 200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip*R_tc)
            str_attitude = "CAMERA Attitude r=%4.0f  p=%4.0f  y=%4.0f"%(math.degrees(roll_camera),math.degrees(pitch_camera),
                               math.degrees(yaw_camera))
            cv2.putText(self.frame, str_attitude, (0, 250), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            self.CAM_X = pos_camera[0]
            self.CAM_Z = pos_camera[2]
            self.CAM_PITCH = math.degrees(pitch_camera)
        else:
            self.cant_find_mark = True
            cv2.putText(self.frame, "can't find any marker", (0,100), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
 
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    #assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

if __name__ == '__main__':
    rospy.init_node('kw_aruco', anonymous=True)
    marker = ArUco()
    rospy.spin()
