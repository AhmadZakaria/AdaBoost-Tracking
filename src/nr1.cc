/*
 * nr1.cc
 *
 *  Created on: May 5, 2014
 *      Author: richard
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdlib.h>

using namespace cv;


void detectAndDisplay( Mat frame );
CascadeClassifier face_cascade;

int main(int argc, const char* argv[]) {
    // implement your solution for task 1 here
    if (argc != 3) {
        std::cout << "usage: " << argv[0] << " <model> <image>" << std::endl;
        exit(1);
    }
    Mat image;
    // Read the file as colored so we can draw blue Rects laters
    image = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    std::cout<<"Image size: "<< image.rows<< "X"<<image.cols<<std::endl;
    if( !face_cascade.load( argv[1] ) ){ printf("--(!)Error loading\n"); return -1; };
    detectAndDisplay(image);
    waitKey();

    return 0;
}


/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30), Size(100, 100) );

    for( size_t i = 0; i < faces.size(); i++ )
    {
        rectangle(frame, faces[i],Scalar(255,0,0));
        rectangle(frame_gray, faces[i],Scalar(255,0,0));
    }

    // so we can concatenate
    cvtColor( frame_gray, frame_gray, CV_GRAY2BGR );

    //-- Show what you got
    Mat out;
    hconcat(frame,frame_gray,out);
    imshow( "Before and after histogram equalization", out );
}
