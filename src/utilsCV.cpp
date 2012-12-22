/*
 * utilsCV.cpp
 *
 *  Created on: 18/12/2012
 *      Author: xescriche
 */
#include <cstdio>
#include <iostream>
#include <math.h>
#include <string.h>
#include "utilsCV.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;

bool readImage(const string& imageName, Mat& image,int color) {
	image = imread(imageName, color);
	if (image.empty()) {
		cout << "The image can not be read." << endl << ">" << endl;
		return false;
	}
	return true;
}

bool readImagesFromFile(const string& imagesFilename,vector <Mat>& imagesVector, vector<string>& imagesVectorNames, int color/*, int & numTrainImages*/) {
	string trainDirName;
	readVocabularyImages(imagesFilename, trainDirName, imagesVectorNames);
	if (imagesVectorNames.empty()) {
		cout << "The images cannot be read." << endl << ">" << endl;
		return false;
	}
	int readImageCount = 0;
	for (size_t i = 0; i < imagesVectorNames.size(); i++) {
		string filename = trainDirName + imagesVectorNames[i];
		Mat img = imread(filename, color);
		if (img.empty())
			cout << "The image " << filename << " cannot be read." << endl;
		else
			readImageCount++;
		imagesVector.push_back(img);
	}
	if (!readImageCount) {
		cout << "All the file images cannot be read." << endl << ">" << endl;
		return false;
	} else {
//		numTrainImages = readImageCount;
		cout << "Number of images from file:                        " << readImageCount << endl;
	}
    return true;
}

void detectKeypointsImage(const Mat& image, vector<KeyPoint>& imageKeypoints, Ptr<FeatureDetector>& featureDetector) {
        featureDetector->detect(image, imageKeypoints);
}

void detectKeypointsImagesVector(const vector<Mat>& imagesVector, vector<vector<KeyPoint> >& imageKeypointsVector,Ptr<FeatureDetector>& featureDetector) {
        featureDetector->detect(imagesVector, imageKeypointsVector);
}

void computeDescriptorsImage(const Mat& image, vector<KeyPoint>& imageKeypointsVector, Mat& imageDescriptors, Ptr<DescriptorExtractor>& descriptorExtractor/*, int & numImageDescriptors*/) {
        descriptorExtractor->compute(image, imageKeypointsVector, imageDescriptors);
//        numImageDescriptors = imageDescriptors.rows;
//        cout << "Number of query image descriptors:     " << numImageDescriptors << endl;
}

void computeDescriptorsImagesVector(const vector<Mat>& imagesVector, vector<vector<KeyPoint> >& imagesVectorKeypointsVector, vector<Mat>& imagesVectorDescriptors,Ptr<DescriptorExtractor>& descriptorExtractor/*, int & numImagesVectorDescriptors*/) {
        descriptorExtractor->compute(imagesVector, imagesVectorKeypointsVector, imagesVectorDescriptors);
//        int totalTrainDesc = 0;
//        for (vector<Mat>::const_iterator tdIter = imagesVectorDescriptors.begin(); tdIter != imagesVectorDescriptors.end(); tdIter++)
//            totalTrainDesc += tdIter->rows;
//        numImagesVectorDescriptors = totalTrainDesc;
//        cout << "Number of train images descriptors:     " << totalTrainDesc << endl;
}


