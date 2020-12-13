#include "ros/ros.h"  // ROS 기본파일
//#include "std_msgs/String.h"
#include "std_msgs/Bool.h"
#include <stdio.h>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
//#include <sstream>

int getch(void)
{
    struct termios oldattr, newattr;
    int ch;
    tcgetattr( STDIN_FILENO, &oldattr );
    newattr = oldattr;
    newattr.c_lflag &= ~( ICANON | ECHO );
    tcsetattr( STDIN_FILENO, TCSANOW, &newattr );
    ch = getchar();
    tcsetattr( STDIN_FILENO, TCSANOW, &oldattr );
    return ch;
}

/* reads from keypress, echoes */
int getche(void)
{
    struct termios oldattr, newattr;
    int ch;
    tcgetattr( STDIN_FILENO, &oldattr );
    newattr = oldattr;
    newattr.c_lflag &= ~( ICANON );
    tcsetattr( STDIN_FILENO, TCSANOW, &newattr );
    ch = getchar();
    tcsetattr( STDIN_FILENO, TCSANOW, &oldattr );
    return ch;
}
bool kbhit()
{
    termios term;
    tcgetattr(0, &term);

    termios term2 = term;
    term2.c_lflag &= ~ICANON;
    tcsetattr(0, TCSANOW, &term2);

    int byteswaiting;
    ioctl(0, FIONREAD, &byteswaiting);

    tcsetattr(0, TCSANOW, &term);

    return byteswaiting > 0;
}


int main(int argc, char** argv)
{
	ros::init(argc, argv, "tpublisher"); 
	ros::NodeHandle nh; 
	ros::Publisher pub = nh.advertise<std_msgs::Bool>("Go_Cam",1);
	ros::Rate loop_rate(10); 
	//std_msgs::String msg; 
	std_msgs::Bool bool_msg;


	//mg.stamp = ros::Time::now(); //현재 시간을 mg의 stamp 변수에 담는다
	//std::stringstream ss;
	//ss << "ROS2000";
	//msg.data = ss.str();


	bool dat = false;
	int key;
	char c;
	//bool turn_while = true;
	ROS_INFO("send camera set true");
	while(true) 
	{
		if(kbhit()){
			c = getch();
			switch(c)
			{
			case 'c':
				dat = true;
				break;
			default:
				dat = false;
				break;
			}
			//loop.sleep();
		}
		bool_msg.data = dat;
		ROS_INFO("set %d",dat);
		//while(turn_while){
			//int connections = pub.getNumSubscribers();
			//if(connections > 0){
				pub.publish(bool_msg);

				//turn_while = false;
			//}
		//}
	}
	return 0;
}













