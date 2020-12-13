#!/usr/bin/env python
import rospy # ros를 위한 module
import cv2
import cv2.aruco as aruco # opencv_contrib의 aruco 모듈을 사용하였다
import numpy as np
import sys, time, math
# ros 내에서 opencv 연계를 위한 module
from cv_bridge import CvBridge, CvBridgeError

# msgs
from sensor_msgs.msg import CompressedImage # 통신 속도 향상을 위해 compressedimage를 사용한다
from geometry_msgs.msg import Pose2D # float64 사이즈의 x, y, theta
from std_msgs.msg import Bool

myid = 72
marker_size = 4.115 #[cm]

# 로지텍 카메라의 내부 파라미터 <- Calibration 프로그램을 통해 얻어냈다.
cam_mat = np.matrix([[520.388401, 0.000000, 322.289271],
            [0.000000, 519.785609, 252.492932],
            [0.000000, 0.000000, 1.000000]])
cam_distort = np.matrix([0.025544, -0.182058, 0.007314, 0.005738, 0.000000])

font = cv2.FONT_HERSHEY_PLAIN

# x 축 중심 180도 회전행렬
R_flip  = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] =  1.0
R_flip[1,1] = -1.0
R_flip[2,2] = -1.0

aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters  = aruco.DetectorParameters_create()

class ArUco:
    def __init__(self):
        # 통신 타이밍을 맞추기 위한 변수들
        self.freeze_cnt = 0
        self.freeze = False
        self.call_once_switch = True
        self.cant_find_mark = True
        self.gogo = False

        # 마커 기준 카메라의 x, z, pitch 저장
        self.CAM_X = 0.0
        self.CAM_Z = 0.0
        self.CAM_PITCH = 0.0

        # 영상 처리 및 영상 통신을 위한 변수
        self.frame = []
        self.bridge = CvBridge()
        # "usb_cam/image_raw/compressed" 이름을 가진 CompressedImage형태의 메세지를 받을 경우, self.image_callback 함수 실행하는 subscriber 선언
        self.sub_image = rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.image_callback)

        # Bool형의 데이터가 담긴 "Go_Cam" 이름의 메세지를 받을 경우 self.cam_on_callback 함수 실행하는 subcribe 선언
        self.sub = rospy.Subscriber("Go_Cam", Bool, self.cam_on_callback)

        # queue 사이즈가 1000이고 latch이 설정된, Pose2D 메세지를 보낼 "Movement"이름의 publisher 선언
        # queue_size: 보낼 메세지를 쌓아두는 queue 구조체
        # latch: 메세지를 느리지만 확실하게? 보내는 역할
        self.pub = rospy.Publisher("Movement",Pose2D,latch=True,queue_size=1000)

    # callback 함수는 통신이 될 때, while문과 같이 작동하므로 imshow, waitKey를 통해 영상을 동영상으로 송출할 수 있다.
    def image_callback(self, img_msg):
        #cv_bridge를 통해 영상 데이터를 수신
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(img_msg, "passthrough")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error: {0}".format(e))
        self.frame = cv_image
        self.Image_Process()

        # rospy.loginfo("",) : ROS python에서 print와 같은 역할
        rospy.loginfo("x= %4.0f, z= %4.0f, pitch= %4.0f",self.CAM_X,self.CAM_Z,self.CAM_PITCH)
        cv2.imshow("Image Window", self.frame) 
        cv2.waitKey(1)

    # Go_Cam 이름의 메세지를 받았을 때 동작하는 callback 함수: data.data에 bool형 데이터가 담겨온다. 
    def cam_on_callback(self, data):
        p = Pose2D()
        p.x = abs(self.CAM_X) # 카메라의 X 좌표를 Pose2D메세지의 x이름의 메세지 속 변수에 넣는다
        p.y = self.CAM_Z # 카메라의 Z 좌표를 Pose2D메세지의 y이름의 메세지 속 변수에 넣는다
        p.theta = self.CAM_PITCH # pitch 데이터를 theta 이름의 메세지 속 변수에 넣는다.

        freeze_cnt_max = 10
        if data.data == True : # data가 True일 경우
            self.gogo = True
            self.freeze_cnt += 1
            if self.freeze_cnt > freeze_cnt_max and self.call_once_switch and not self.cant_find_mark:
                #self.freeze = True
                self.pub.publish(p) # p 변수에 담긴 Pose2D 메세지를 보낸다. 어떤 설정으로 어떤 이름으로 보내는 지는 publish 선언 참고
                rospy.loginfo("I send x= %4.0f, z= %4.0f, pitch= %4.0f",abs(self.CAM_X),self.CAM_Z,self.CAM_PITCH)
                self.call_once_switch = False
            elif self.freeze_cnt <= freeze_cnt_max and self.call_once_switch and not self.cant_find_mark:
                rospy.loginfo("publish countdown = %d/%d ",self.freeze_cnt, freeze_cnt_max)
            elif self.freeze_cnt > freeze_cnt_max and self.call_once_switch and self.cant_find_mark:
                rospy.loginfo("there is no marker. please show me one")
        elif data.data == False: # data가 False일 경우
            self.gogo = False
            self.freeze_cnt = 0
            self.call_once_switch = True
            #self.freeze = False

        # publish와 subscribe는 while문 속에서 동작한다.( 메세지 한 번 보내는 것으로 확실하게 전달되지 않기 때문? )
        # get_num_connections함수를 통해 연결될 때 단 한번만 publish 하는 역할을 하는 알고리즘
        #on_while = True
        #while on_while:
            #connections = self.pub.get_num_connections()
            #if connections > 0:
                #self.pub.publish(p)
                #rospy.loginfo("send Pose2D")
                #on_while = False


    # 영상 처리 함수
    def Image_Process(self):
        ret = self.frame
        gray    = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) # 영상을 grayscale로 변환
        # 카메라 파라미터, grayscale이미지, dictionary와 파라미터를 참고하여 마커를 찾는다. 모서리와 마커 id 반환 (rejected?)
        corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters,
                              cameraMatrix=cam_mat, distCoeff=cam_distort)
        # 마커가 검출될 경우
        if ids is not None : #and self.freeze is not True:
            self.cant_find_mark = False
            # corner, 마커 크기, 카메라 파라미터를 이용하여 마커의 pose를 알아낸다
            ret = aruco.estimatePoseSingleMarkers(corners, marker_size, cam_mat, cam_distort)
            rvec, tvec = ret[0][0,0,:], ret[1][0,0,:] # 알아낸 pose 행렬을 자세, 위치 벡터로 변환
            aruco.drawDetectedMarkers(self.frame, corners) # 코너를 이용하여 frame에 마커를 그린다
            aruco.drawAxis(self.frame, cam_mat, cam_distort, rvec, tvec, 10) # 카메라 파라미터, 마커 위치 및 자세 벡터를 이용하여 frame 이미지에 10 길이의 xyz축을 그린다. 

            str_position = "MARKER Position x=%4.0f  y=%4.0f  z=%4.0f"%(tvec[0], tvec[1], tvec[2]) # 카메라 기준 마커의 위치 좌표
            cv2.putText(self.frame, str_position, (0, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            R_ct    = np.matrix(cv2.Rodrigues(rvec)[0]) # 로드리그스 행렬을 이용하여 벡터를 회전 행렬 변환
            R_tc    = R_ct.T

            roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip*R_tc) # 오일러 행렬로 변환
            str_attitude = "MARKER Attitude r=%4.0f  p=%4.0f  y=%4.0f"%(math.degrees(roll_marker),math.degrees(pitch_marker),math.degrees(yaw_marker)) # 마커의 roll, pitch, yaw
            cv2.putText(self.frame, str_attitude, (0, 150), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            pos_camera = -R_tc*np.matrix(tvec).T # 회전 행렬을 이용하여 마커 기준 카메라의 위치 벡터 반환
            
            str_position = "CAMERA Position x=%4.0f  y=%4.0f  z=%4.0f"%(pos_camera[0], pos_camera[1], pos_camera[2]) # 마커 기준 카메라의 위치
            cv2.putText(self.frame, str_position, (0, 200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # 마커 기준 카메라의 자세는 카메라 기준 마커의 자세와 같다
            str_attitude = "CAMERA Attitude r=%4.0f  p=%4.0f  y=%4.0f"%(math.degrees(roll_marker),math.degrees(pitch_marker), math.degrees(yaw_marker))
            cv2.putText(self.frame, str_attitude, (0, 250), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            self.CAM_X = pos_camera[0]
            self.CAM_Z = pos_camera[2]
            self.CAM_PITCH = math.degrees(pitch_camera)
        else: # 마커가 검출되지 않았을 경우
            self.cant_find_mark = True
            cv2.putText(self.frame, "can't find any marker", (0,100), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

# 회전행렬 검사 함수
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# 오일러 행렬 변환 함수
def rotationMatrixToEulerAngles(R):
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
    rospy.spin() # ros spin()함수를 통해 (돌면서) 수시로 메세지가 왔는지 확인한다. 이를 통해 subscriber가 동작한다.
