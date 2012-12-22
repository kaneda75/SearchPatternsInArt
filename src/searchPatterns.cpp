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
#include "utils.hpp"
#include "utilsCV.hpp"

using namespace cv;
using namespace std;

const string vocabularyImagesNameFile = "/Users/xescriche/git/SearchPattersInArt/tests/test1/vocabularyImages.txt";
const string newImageFileName = "/Users/xescriche/git/SearchPatternsInArt/tests/test1/tapies3.jpg";
const string detectorType = "SIFT";
const string descriptorType = "SIFT";
const int color = 0;
const string dirToSaveResImages = "/Users/xescriche/git/SearchPatternsInArt/tests/test1/results";


void computeMatching() {
	try {
		// SIFT Feature detector
		Ptr<FeatureDetector> featureDetector;
		featureDetector = FeatureDetector::create(detectorType);
		if (featureDetector.empty()) {
			cout << "The detector cannot be created." << endl << ">" << endl;
		} else {
			cout << "Detector: " << detectorType << endl;
		}

		// SIFT Descriptor extractor
		Ptr<DescriptorExtractor> descriptorExtractor;
		descriptorExtractor = DescriptorExtractor::create(descriptorType);
		if (featureDetector.empty()) {
			cout << "The descriptor cannot be created." << endl << ">" << endl;
		} else {
			cout << "Descriptor: " << descriptorType << endl;
		}

		// Vocabulary images
		vector<Mat> vocabularyImages;
		vector<string> vocabularyImagesNames;
		if (!readImagesFromFile(vocabularyImagesNameFile, vocabularyImages,vocabularyImagesNames, color)) {
			cout << endl;
		}

		vector<vector<KeyPoint> > vocabularyImagesKeypoints;
		detectKeypointsImagesVector(vocabularyImages, vocabularyImagesKeypoints, featureDetector);

		vector<Mat> imagesVectorDescriptors;
		computeDescriptorsImagesVector(vocabularyImages, vocabularyImagesKeypoints,imagesVectorDescriptors, descriptorExtractor);


		// New Image
//		Mat newImage;
//		if (!readImage(newImageFileName,newImage,color)) {
//			cout << endl;
//		}
//
//		vector<KeyPoint> queryKeypoints;
//		detectKeypoints(newImage, queryKeypoints, vocabularyImages,trainKeypoints, featureDetector);
//
//		Mat queryDescriptors;
//		computeDescriptors(newImage, vocabularyImagesKeypoints,queryDescriptors, vocabularyImages, trainKeypoints,trainDescriptors, descriptorExtractor,numQueryDescriptors, numTrainDescriptors);

	} catch (exception& e) {
		cout << e.what() << endl;
	}
}

int main(int argc, char *argv[]) {
	computeMatching();
}
