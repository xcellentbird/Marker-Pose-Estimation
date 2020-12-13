# Marker-Pose-Estimation
개발 환경:
Linux Ubuntu 18.04 ROS-Melodic

kw_aruco_py
- 컴퓨터 비전 노드: 마커 검출 및 위치, 자세 추정
- 의존 패키지: cv_bridge, geometry_msgs, image_transfer, message_generation, rospy, sesnsor_msgs, std_msgs

// topictest, topictest2는 kw_aruco_py를 사용하기 위한 예시 통신 노드입니다.
topictest - tpublisher
- publish node: 'c'키 입력으로 kw_aruco_py 패키지에 전송 on 메세지 publish

topictest2 - tsubscriber
- subscribe node: kw_aruco_py 패키지로부터 마커의 위치 및 자세 데이터를 subscribe

노드 그래프
![rosgraph2](https://user-images.githubusercontent.com/59414764/102018984-f2fc6680-3db3-11eb-9522-5a7c702170cf.png)

실행 화면 스크린샷
![스크린샷, 2020-12-14 01-58-10](https://user-images.githubusercontent.com/59414764/102019140-fe9c5d00-3db4-11eb-8f27-78c6788b3613.png)
