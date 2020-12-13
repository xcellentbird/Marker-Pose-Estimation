#include "ros/ros.h"  
#include "geometry_msgs/Pose2D.h"

int cnt = 1;
// Pose2D 메세지를 받는 콜백함수
void msgCallBack(const geometry_msgs::Pose2D& msg)
{
	ROS_INFO("recieve %d th pose msg = x= %4.0f, z= %4.0f, p= %4.0f ",cnt,msg.x, msg.y, msg.theta);
	cnt++;
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "tsubscriber"); // 노드 초기화
	ros::NodeHandle nh; // 노드 핸들러 설정
	ros::Subscriber sub = nh.subscribe("Movement",1,msgCallBack); // subscriber 설정 및 콜백 함수 설정
	ROS_INFO("ready to receive Pose2D");
	ros::spin(); // spin - subscriber를 동작시키는 역할
	return 0;
}
	
