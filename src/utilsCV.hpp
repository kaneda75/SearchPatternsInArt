/*
 * utilsCV.hpp
 *
 *  Created on: 18/12/2012
 *      Author: xescriche
 */
#ifndef UTILSCV_HPP_
#define UTILSCV_HPP_

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;
using namespace cv;

bool readImage(const string& imageName, Mat& image,int color);
bool readImagesFromFile(const string& imagesFilename,vector <Mat>& imagesVector, vector<string>& imagesVectorNames, int color,int& numImagesTotal);
void computeDescriptorsImage(const Mat& image, vector<KeyPoint>& imageKeypointsVector, Mat& imageDescriptors, Ptr<DescriptorExtractor>& descriptorExtractor);
void computeDescriptorsImagesVector(const vector<Mat>& imagesVector, vector<vector<KeyPoint> >& imagesVectorKeypointsVector, vector<Mat>& imagesVectorDescriptors,Ptr<DescriptorExtractor>& descriptorExtractor);
void detectKeypointsImage(const Mat& image, vector<KeyPoint>& imageKeypoints, Ptr<FeatureDetector>& featureDetector);
void detectKeypointsImagesVector(const vector<Mat>& imagesVector, vector<vector<KeyPoint> >& imageKeypointsVector,Ptr<FeatureDetector>& featureDetector);
void showKeypointsImagesVector(const vector<Mat>& vocabularyImages, const vector<vector<KeyPoint> >& vocabularyImagesKeypoints);
void showKeypointsImage(const Mat& image, const vector<KeyPoint> & imageKeypoints);
void kmeansVocabularyImages(const vector<Mat>& imagesVectorDescriptors, int clusterCount, int criteriaKMeans, int attemptsKMeans, int numImagesTotal, vector<vector<int> >& vocabulary, Mat& labels, Mat& centers, int numRowsTotal);
void findKCentersOnNewImage(Mat& matCenters, Mat& newImageDescriptors, Mat& centers);
Mat votingImages(vector<vector<int> >& vocabulary,Mat& matCenters, int numImagesTotal);
void showMatrixValues2(Mat& matrix, string s);
void showMatrixValues3(vector<KeyPoint> keypoints, Mat& matrix,  string s);
int calculeNumRowsTotal(const vector<Mat>& imagesVectorDescriptors);
int getMostVotedImage(Mat matVote);
void ransac(const Mat& wordsImageIni,const Mat& wordsNewImage, Mat imageIni,const vector<KeyPoint>& imageIniKeypoints, Mat newImage,const vector<KeyPoint>& newImageKeypoints, int clusterCount, const string dirToSaveResImages, int imag, int thresholdDistanceAdmitted);


#endif /* UTILSCV_HPP_ */
