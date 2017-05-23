/*
 * nr3.cc
 *
 *  Created on: Apr 28, 2014
 *      Author: richard
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
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
#define _displacement_x 30 //120
#define _displacement_y 15 //60

void computeHistogram(const cv::Mat& image, const cv::Point& p, Vector& histogram) {

}

void generateTrainingData(std::vector<Example>& data, const std::vector<cv::Mat>& imageSequence, const std::vector<cv::Point>& referencePoints) {

}

void loadImage(const std::string& imageFile, cv::Mat& image) {

}

void loadTrainFrames(const char* trainDataFile, std::vector<cv::Mat>& imageSequence,
		std::vector<cv::Point>& referencePoints) {

}

void loadTestFrames(const char* testDataFile, std::vector<cv::Mat>& imageSequence, cv::Point& startingPoint) {

}

void findBestMatch(const cv::Mat& image, cv::Point& lastPosition, AdaBoost& adaBoost) {

}

void drawTrackedFrame(cv::Mat& image, cv::Point& position) {
	cv::rectangle(image, cv::Point(position.x - _objectWindow_width / 2, position.y - _objectWindow_height / 2),
			cv::Point(position.x + _objectWindow_width / 2, position.y + _objectWindow_height / 2), 0, 3);
	cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE);
	cv::imshow("Display window", image);
	//std::sleep(1);
	cv::waitKey(0);
}

int main( int argc, char** argv )
{

	//implement the functions above to make this work. Or implement your solution from scratch
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

	return 0;
}
