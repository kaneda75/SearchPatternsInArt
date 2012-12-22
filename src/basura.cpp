///*
// * basura.cpp
// *
// *  Created on: 18/12/2012
// *      Author: xescriche
// */
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/contrib/contrib.hpp"
//#include <iostream>
//#include <fstream>
//#include <vector>
//#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>
//#include <sstream>
//
//using namespace cv;
//using namespace std;
//
//static void maskMatchesByTrainImgIdx(const vector<DMatch>& matches, int trainImgIdx, vector<char>& mask) {
//        mask.resize(matches.size());
//        fill(mask.begin(), mask.end(), 0);
//        for (size_t i = 0; i < matches.size(); i++) {
//            if (matches[i].imgIdx == trainImgIdx)
//                mask[i] = 1;
//        }
//}
//
//
//static bool createDetectorDescriptorMatcher(const string& detectorType, const string& descriptorType, const string& matcherType,Ptr<FeatureDetector>& featureDetector,Ptr<DescriptorExtractor>& descriptorExtractor,Ptr<DescriptorMatcher>& descriptorMatcher)  {
//        featureDetector = FeatureDetector::create(detectorType);
//        descriptorExtractor = DescriptorExtractor::create(descriptorType);
//        descriptorMatcher = DescriptorMatcher::create(matcherType);
//
//    bool isCreated = !(featureDetector.empty() || descriptorExtractor.empty() || descriptorMatcher.empty());
//    if (!isCreated)
//        cout << "No es pot crear el detector, el descriptor o el matcher." << endl << ">" << endl;
//    return isCreated;
//}
//
//void readFileDetectors(string detectors[NUM_DETECTORS]) {
//	FILE* ftxt;
//	char linea[120] = "\0";
//	int error2 = 0;
//	int i = 0;
//	char file[120];
//	sprintf(file, DETECTORS_TYPE_FILE);
//	ftxt = fopen(file, "rt");
//	if (ftxt != NULL) {
//		error2 = leerLineaTxt(ftxt, linea);
//		while ((!feof(ftxt)) && (!error2)) {
//			detectors[i] = linea;
//			detectors[i].erase(detectors[i].length() - 1, 1);
//			i++;
//			if (!error2)
//				leerLineaTxt(ftxt, linea);
//		}
//		fclose(ftxt);
//	}
//}
//
//void readFileDetescriptors(string descriptors[NUM_DESCRIPTORS]) {
//	FILE* ftxt;
//	char linea[120] = "\0";
//	int error2 = 0;
//	int i = 0;
//	char file[120];
//	sprintf(file, DESCRIPTORS_TYPE_FILE);
//	ftxt = fopen(file, "rt");
//	if (ftxt != NULL) {
//		error2 = leerLineaTxt(ftxt, linea);
//		while ((!feof(ftxt)) && (!error2)) {
//			descriptors[i] = linea;
//			descriptors[i].erase(descriptors[i].length() - 1, 1);
//			i++;
//			if (!error2)
//				leerLineaTxt(ftxt, linea);
//		}
//		fclose(ftxt);
//	}
//}
//
//void readFileMatchers(string matchers[NUM_MATCHERS]) {
//	FILE* ftxt;
//	char linea[120] = "\0";
//	int error2 = 0;
//	int i = 0;
//	char file[120];
//	sprintf(file, MATCHERS_TYPE_FILE);
//	ftxt = fopen(file, "rt");
//	if (ftxt != NULL) {
//		error2 = leerLineaTxt(ftxt, linea);
//		while ((!feof(ftxt)) && (!error2)) {
//			matchers[i] = linea;
//			matchers[i].erase(matchers[i].length() - 1, 1);
//			i++;
//			if (!error2)
//				leerLineaTxt(ftxt, linea);
//		}
//		fclose(ftxt);
//	}
//}
//
//
//static void computeDescriptors(const Mat& queryImage, vector<KeyPoint>& queryKeypoints, Mat& queryDescriptors,const vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints, vector<Mat>& trainDescriptors,Ptr<DescriptorExtractor>& descriptorExtractor, int & numQueryDescriptors, int & numTrainDescriptors) {
//        descriptorExtractor->compute(queryImage, queryKeypoints, queryDescriptors);
//        descriptorExtractor->compute(trainImages, trainKeypoints, trainDescriptors);
//        int totalTrainDesc = 0;
//        for (vector<Mat>::const_iterator tdIter = trainDescriptors.begin(); tdIter != trainDescriptors.end(); tdIter++)
//            totalTrainDesc += tdIter->rows;
//
//        numQueryDescriptors = queryDescriptors.rows;
//        cout << "Number of query image descriptors:     " << numQueryDescriptors << endl;
//        numTrainDescriptors = totalTrainDesc;
//        cout << "Number of train images descriptors:     " << totalTrainDesc << endl;
//}
//
//static void matchDescriptors(const Mat& queryDescriptors, const vector<Mat>& trainDescriptors, vector<DMatch>& matches, Ptr<DescriptorMatcher>& descriptorMatcher, int & numMatches, double & matchTime) {
//        TickMeter tm;
//        tm.start();
//        descriptorMatcher->add(trainDescriptors);
//        descriptorMatcher->train();
//        tm.stop();
//        tm.start();
//        descriptorMatcher->match(queryDescriptors, matches);
//        tm.stop();
//        matchTime = tm.getTimeMilli();
//        CV_Assert(queryDescriptors.rows == (int) matches.size() || matches.empty());
//        numMatches = matches.size();
//        cout << "Number of matches:                              " << numMatches << endl;
//        cout << "Match time:                                            " << matchTime << " ms" << endl;
//}
//
//
//static void saveResultImage(const Mat& image, const vector<KeyPoint>& imageKeypoints, const string& resultDir, string nomArxiu) {
//        Mat drawImg;
//        vector<char> mask;
//        for (size_t i = 0; i < trainImages.size(); i++) {
//            if (!trainImages[i].empty()) {
////                maskMatchesByTrainImgIdx(matches, (int) i, mask);
//                drawMatches(image, imageKeypoints, trainImages[i], trainKeypoints[i],
//                        matches, drawImg, Scalar(255, 0, 0), Scalar(0, 255, 255), mask);
//                string filename = resultDir + "/" + nomArxiu + "_" + trainImagesNames[i];
//                if (!imwrite(filename, drawImg))
//                    cout << "L'imatge " << filename << " no pot ser guardada (pot ser que el directori " << resultDir << " no existeixi)." << endl;
//            }
//        }
//}
//
//static void saveResultImage(const Mat& queryImage, const vector<KeyPoint>& queryKeypoints,const vector<Mat>& trainImages, const vector<vector<KeyPoint> >& trainKeypoints, const vector<DMatch>& matches, const vector<string>& trainImagesNames, const string& resultDir, string nomArxiu) {
//        Mat drawImg;
//        vector<char> mask;
//        for (size_t i = 0; i < trainImages.size(); i++) {
//            if (!trainImages[i].empty()) {
////                maskMatchesByTrainImgIdx(matches, (int) i, mask);
//                drawMatches(queryImage, queryKeypoints, trainImages[i], trainKeypoints[i],
//                        matches, drawImg, Scalar(255, 0, 0), Scalar(0, 255, 255), mask);
//                string filename = resultDir + "/" + nomArxiu + "_" + trainImagesNames[i];
//                if (!imwrite(filename, drawImg))
//                    cout << "L'imatge " << filename << " no pot ser guardada (pot ser que el directori " << resultDir << " no existeixi)." << endl;
//            }
//        }
//}
