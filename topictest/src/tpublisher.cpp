#include "ros/ros.h"  // ROS 기본파일
//#include "std_msgs/String.h"
#include "std_msgs/Bool.h"
#include <stdio.h>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
//#include <sstream>

// 입력된 키보드의 char를 반환
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

// 키보드 입력을 확인하는 함수
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
	ros::NodeHandle nh; // 노드 핸들러 선언 ( 참고로 노드 핸들러 여러 개를 선언할 수 있다 )
	ros::Publisher pub = nh.advertise<std_msgs::Bool>("Go_Cam",1); // publiser 선언
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
		// 키보드의 입력이 있을 경우
		if(kbhit()){
			// 입력을 char로 반환
			c = getch();
			switch(c)
			{
			case 'c': // 입력이 c인 경우 dat를 true로 설정
				dat = true;
				break;
			default: // 다른 입력의 경우 dat를 false로 설정
				dat = false;
				break;
			}
			//loop.sleep();
		}
		bool_msg.data = dat; // dat를 메세지 data형에 넣는다
		ROS_INFO("set %d",dat); // printf와 같은 역할
		//while(turn_while){
			//int connections = pub.getNumSubscribers();
			//if(connections > 0){
				pub.publish(bool_msg); // bool_msg를 publish

				//turn_while = false;
			//}
		//}
	}
	return 0;
}













