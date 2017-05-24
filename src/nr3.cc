/*
 * nr3.cc
 *
 *  Created on: Apr 28, 2014
 *      Author: richard
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <sstream>
#include "Types.hh"
#include "AdaBoost.hh"

#define _objectWindow_width 121
#define _objectWindow_height 61

#define _searchWindow_width 61
#define _searchWindow_height 61

// use 30/15 for overlapping negative examples and 120/60 for non-overlapping negative examples
#define _displacement_x 120 //30 //120
#define _displacement_y 60 //15 //60

void computeHistogram(const cv::Mat& image, const cv::Point& p, Vector& histogram) {
    u32 minx = p.x - (_objectWindow_width / 2.0f);
    u32 miny = p.y - (_objectWindow_height / 2.0f);
    cv::Rect roi(minx, miny, _objectWindow_width, _objectWindow_height);

    // make sure it doesn't go out of bounds
    roi =  roi & cv::Rect(0, 0, image.cols, image.rows);
    cv::Mat roiMat = image(roi);

    histogram.resize(256);
    std::fill(histogram.begin(), histogram.end(), 0);
    // normalized increment
    f32 increment = 1.0f / (roiMat.cols * roiMat.rows);
    for (int row = 0; row < roiMat.rows; ++row) {
        for (int col = 0; col < roiMat.cols; ++col) {
            u8 idx = roiMat.at<u8>(row,col);
            histogram[idx]+=increment;
        }
    }
}

void generateTrainingData(std::vector<Example>& data, const std::vector<cv::Mat>& imageSequence, const std::vector<cv::Point>& referencePoints) {
    // for each image sequence (frame) and reference pt
    for (int i = 0; i < imageSequence.size(); ++i) {
        // // compute hist around point itself
        // // make and append example with hist and label 0
        Example ePos;
        ePos.label = 1;
        computeHistogram(imageSequence[i], referencePoints[i], ePos.attributes);
        data.push_back(ePos);

        Example eNegTL;//top left
        eNegTL.label = 0;
        cv::Point topLeft(referencePoints[i].x - _displacement_x, referencePoints[i].y - _displacement_y);
        computeHistogram(imageSequence[i], topLeft, eNegTL.attributes);
        data.push_back(eNegTL);

        Example eNegBR;//bottom right
        eNegBR.label = 0;
        cv::Point botRight(referencePoints[i].x + _displacement_x, referencePoints[i].y + _displacement_y);
        computeHistogram(imageSequence[i], botRight, eNegBR.attributes);
        data.push_back(eNegBR);

        Example eNegBL;//bottom left
        eNegBL.label = 0;
        cv::Point botLeft(referencePoints[i].x - _displacement_x, referencePoints[i].y + _displacement_y);
        computeHistogram(imageSequence[i], botLeft, eNegBL.attributes);
        data.push_back(eNegBL);

        Example eNegTR;//Top Right
        eNegTR.label = 0;
        cv::Point topRight(referencePoints[i].x + _displacement_x, referencePoints[i].y - _displacement_y);
        computeHistogram(imageSequence[i], topRight, eNegTR.attributes);
        data.push_back(eNegTR);
    }

}

void loadImage(const std::string& imageFile, cv::Mat& image) {
    std::string path = "nemo/";
    image = cv::imread(path.append(imageFile).c_str(), CV_LOAD_IMAGE_GRAYSCALE);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        std::cerr <<  "Could not open or find the image: " << imageFile << std::endl ;
        exit(-1);
    }
}

void loadTrainFrames(const char* trainDataFile, std::vector<cv::Mat>& imageSequence,
                     std::vector<cv::Point>& referencePoints) {
    std::fstream inFile;
    inFile.open( trainDataFile, std::ios::in ) ;
    std::string frameName;
    int x, y;
    while(inFile>> frameName >> x >> y){
        cv::Mat image;
        loadImage(frameName, image);
        // append to image sequences
        imageSequence.push_back(image);

        cv::Point refPt(x,y);
        referencePoints.push_back(refPt);
    }
    inFile.close( );

}

void loadTestFrames(const char* testDataFile, std::vector<cv::Mat>& imageSequence, cv::Point& startingPoint) {
    std::fstream inFile;
    inFile.open( testDataFile, std::ios::in ) ;
    std::string frameName;

    inFile >> startingPoint.x >>startingPoint.y;
    imageSequence.clear();
    while(inFile>> frameName ){
        cv::Mat image;
        loadImage(frameName, image);
        // append to image sequences
        imageSequence.push_back(image);
    }
    inFile.close( );
}

void drawTrackedFrame(cv::Mat& image, cv::Point& position) {
    cv::rectangle(image, cv::Point(position.x - _objectWindow_width / 2, position.y - _objectWindow_height / 2),
                  cv::Point(position.x + _objectWindow_width / 2, position.y + _objectWindow_height / 2), 0, 3);
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display window", image);
    //std::sleep(1);
    cv::waitKey(1);
}

void findBestMatch(cv::Mat& image, cv::Point& lastPosition, AdaBoost& adaBoost) {
    f32 best_confidence = -1.0f;
    cv::Point bestPoint;
    for (int minx = - (_searchWindow_width / 2); minx <  (_searchWindow_width / 2); ++minx) {
        for (int miny = - (_searchWindow_height / 2); miny < (_searchWindow_height / 2); ++miny) {
            cv::Point p(lastPosition.x + minx, lastPosition.y + miny);
            Vector v;
            computeHistogram(image, p, v);
            u32 ptClass = adaBoost.classify(v);
            if(ptClass == 1){
                f32 conf = adaBoost.confidence(v, 1);
                if(conf > best_confidence){
                    bestPoint.x = p.x;
                    bestPoint.y = p.y;
                    best_confidence = conf;
                }
            }
        }
    }
    if(best_confidence > 0.5)
        lastPosition = bestPoint;
    std::cout << "Tracking confidence "<< best_confidence << std::endl;
}

int main( int argc, char** argv )
{

    if(argc != 4) {
        std::cout <<" Usage: " << argv[0] << " <training-frame-file> <test-frame-file> <# iterations for AdaBoost>" << std::endl;
        return -1;
    }

    u32 adaBoostIterations = atoi(argv[3]);

    // load the training frames
    std::vector<cv::Mat> imageSequence;
    std::vector<cv::Point> referencePoints;
    loadTrainFrames(argv[1], imageSequence, referencePoints);

    // generate gray-scale histograms from the training frames:
    // one positive example per frame (_objectWindow_width x _objectWindow_height window around reference point for object)
    // four negative examples per frame (with _displacement_{x/y} + small random displacement from reference point)
    std::vector<Example> trainingData;
    generateTrainingData(trainingData, imageSequence, referencePoints);

    // initialize AdaBoost and train a cascade with the extracted training data
    AdaBoost adaBoost(adaBoostIterations);
    adaBoost.initialize(trainingData);
    adaBoost.trainCascade(trainingData);

    // log error rate on training set
    u32 nClassificationErrors = 0;
    for (u32 i = 0; i < trainingData.size(); i++) {
        u32 label = adaBoost.classify(trainingData.at(i).attributes);
        nClassificationErrors += (label == trainingData.at(i).label ? 0 : 1);
    }
    std::cout << "Error rate on training set: " << (f32)nClassificationErrors / (f32)trainingData.size() << std::endl;

    // load the test frames and the starting position for tracking
    std::vector<Example> testImages;
    cv::Point lastPosition;
    loadTestFrames(argv[2], imageSequence, lastPosition);

    // for each frame...
    for (u32 i = 0; i < imageSequence.size(); i++) {
        // ... find the best match in a window of size
        // _searchWindow_width x _searchWindow_height around the last tracked position
        findBestMatch(imageSequence.at(i), lastPosition, adaBoost);

        // draw the result
        drawTrackedFrame(imageSequence.at(i), lastPosition);
    }
    cv::waitKey(-1);
    return 0;
}
