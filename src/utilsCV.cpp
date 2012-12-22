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

using namespace std;
using namespace cv;

static void detectKeypoints(const Mat& queryImage, vector<KeyPoint>& queryKeypoints,const vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints,Ptr<FeatureDetector>& featureDetector) {
        featureDetector->detect(queryImage, queryKeypoints);
        featureDetector->detect(trainImages, trainKeypoints);

}



