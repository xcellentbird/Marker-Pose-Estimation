cmake_minimum_required(VERSION 3.0.2)
project(topictest)


## 의존성 패키지를 설정하는 구간이다.
find_package(catkin REQUIRED COMPONENTS
  message_generation
  roscpp
  std_msgs
)

## 의존하는 메세지를 설정하는 옵션이다
generate_messages(
  DEPENDENCIES
  std_msgs
)

## LIBRARIES: 프로젝트에 필요한 라이브러리를 만든다
## CATKIN_DEPENDS: 빌드 의존하는 패키지를 넣는다
catkin_package(
 LIBRARIES topictest
 CATKIN_DEPENDS roscpp std_msgs
)

## Specify additional locations of header files
## 인클루드 디렉토리를 설정한다 include를 넣고 라이브러리를 include파일에 넣으면 그 라이브러리를 참고하게 된다
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

## 두번째 인자의 파일을 참고해서 노드를 만들 노드를 설정한다
add_executable(tpublisher src/tpublisher.cpp)
## 첫번째 인자의 노드를 만들 때 의존이 필요한 파일들을 설정한다.
add_dependencies(tpublisher ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
## 첫번째 인자의 노드를 만들 때 참고할 라이브러리 링크를 설정한다.
target_link_libraries(tpublisher ${catkin_LIBRARIES})