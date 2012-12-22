/*
 * utilsCV.hpp
 *
 *  Created on: 18/12/2012
 *      Author: xescriche
 */
#ifndef UTILSCV_HPP_
#define UTILSCV_HPP_

#include <cstdio>
#include <iostream>
#include <math.h>
#include <string.h>
#include "utilsCV.hpp"

using namespace std;
using namespace cv;


static void detectKeypoints(const Mat& queryImage, vector<KeyPoint>& queryKeypoints,const vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints,Ptr<FeatureDetector>& featureDetector);

#endif /* UTILSCV_HPP_ */
