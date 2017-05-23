# Computer Vision Lecture MA-INF 2201
#
# CMakeLists.txt for sheet 03
# see http://www.cmake.org/ for a documentation about CMake
#
# compile your programm using the following commands:
#
# Enter build folder
# $ cd build
#
# !!! To Make an Eclipse Project you should use the Generator in the following way
# .../Sheet03/build/$ cmake -G"Eclipse CDT4 - Unix Makefiles" ../
# .../Sheet03/build/$ make

# OR

# !!! To Make an ordinary Unix Makefile project you can use the Generator in the following way
# .../Sheet03/build/$ cmake -G"Unix Makefiles" ../
# .../Sheet03/build/$ make

# !!! NOTE
# With the way described above you will be sure that 
# created cmake files  will be stored in build folder ,and 
# binary files will be stored in the bin folder
# which you should not version while using the SVN

# Please, When you test your program from terminal after compilation
# switch back to main exercise folder and run your program as following
# .../Sheet03/$ ./bin/sheet03 images/bonn.png  

cmake_minimum_required(VERSION 2.6)
project (sheet03)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
SET(CMAKE_CXX_FLAGS "-std=c++0x -fopenmp")				# New C11
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)			# Binary file location
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})			# Exectuable file location for main and subprojects 
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)		# Library file location for main and subprojects
set(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)			# Source folder
set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src)		# Header folder

set(CMAKE_BUILD_TYPE debug)					# choose if you want to compile with or without debugging flags,
#set(CMAKE_BUILD_TYPE release)					# uncomment only one of the following two commands at a time

#To build the project openCV is required
find_package( OpenCV REQUIRED )

#Include Required Directories
include_directories(${PROJECT_INCLUDE_DIR})			
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

# Create a shared library (To use library creation uncomment the following)
#	add_library(	
#		LIBRARY_NAME 
#		${PROJECT_SOURCE_DIR}/some.cpp
#		${PROJECT_SOURCE_DIR}/someMore.cpp
#		)

#Create an Executable for segment
add_executable(sheet03 ${PROJECT_SOURCE_DIR}/sheet03.cpp)	#Add executable to sheet03.cpp
target_link_libraries( sheet03 ${OpenCV_LIBS} )			#Link the executable with opencv library
add_executable(nr1 ${PROJECT_SOURCE_DIR}/nr1.cc)       #Add executable to sheet03.cpp
target_link_libraries( nr1 ${OpenCV_LIBS} )                 #Link the executable with opencv library
add_executable(nr2 ${PROJECT_SOURCE_DIR}/nr2.cc
                   ${PROJECT_SOURCE_DIR}/AdaBoost.hh
                   ${PROJECT_SOURCE_DIR}/AdaBoost.cc
                   ${PROJECT_SOURCE_DIR}/WeakClassifier.hh
                   ${PROJECT_SOURCE_DIR}/WeakClassifier.cc
                   ${PROJECT_SOURCE_DIR}/Types.hh)       #Add executable to sheet03.cpp
target_link_libraries( nr2 ${OpenCV_LIBS} )                 #Link the executable with opencv library

#(If you want to add subprojects to your main the following is how you do)
#Continue with subdirectory 
# add_subdirectory(SubFolder/That/Contains/Another/CMakeLists.txt)

	

