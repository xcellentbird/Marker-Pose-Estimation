#include "ros/ros.h"  
#include "geometry_msgs/Pose2D.h"

int cnt = 1;
void msgCallBack(const geometry_msgs::Pose2D& msg)
{
	ROS_INFO("recieve %d th pose msg = x= %4.0f, z= %4.0f, p= %4.0f ",cnt,msg.x, msg.y, msg.theta);
	cnt++;
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "tsubscriber"); 
	ros::NodeHandle nh; 
	ros::Subscriber sub = nh.subscribe("Movement",1,msgCallBack);
	ROS_INFO("ready to receive Pose2D");
	ros::spin();
	return 0;
}
	