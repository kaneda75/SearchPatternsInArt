/*
 * basura.cpp
 *
 *  Created on: 18/12/2012
 *      Author: xescriche
 */
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>

using namespace cv;
using namespace std;

static void maskMatchesByTrainImgIdx(const vector<DMatch>& matches, int trainImgIdx, vector<char>& mask) {
        mask.resize(matches.size());
        fill(mask.begin(), mask.end(), 0);
        for (size_t i = 0; i < matches.size(); i++) {
            if (matches[i].imgIdx == trainImgIdx)
                mask[i] = 1;
        }
}


static bool createDetectorDescriptorMatcher(const string& detectorType, const string& descriptorType, const string& matcherType,Ptr<FeatureDetector>& featureDetector,Ptr<DescriptorExtractor>& descriptorExtractor,Ptr<DescriptorMatcher>& descriptorMatcher)  {
        featureDetector = FeatureDetector::create(detectorType);
        descriptorExtractor = DescriptorExtractor::create(descriptorType);
        descriptorMatcher = DescriptorMatcher::create(matcherType);

    bool isCreated = !(featureDetector.empty() || descriptorExtractor.empty() || descriptorMatcher.empty());
    if (!isCreated)
        cout << "No es pot crear el detector, el descriptor o el matcher." << endl << ">" << endl;
    return isCreated;
}
