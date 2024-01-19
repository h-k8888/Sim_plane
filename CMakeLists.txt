cmake_minimum_required(VERSION 2.8.3)
project(sim_plane)

#SET(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -pthread")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")

ADD_COMPILE_OPTIONS(-std=c++14 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set( CMAKE_CXX_FLAGS "-std=c++14 -O3" )

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -std=c++0x -std=c++14 -fexceptions")

find_package(Eigen3 REQUIRED)

include_directories(
        ${EIGEN3_INCLUDE_DIR}
)
add_executable(sim_plane main.cpp)