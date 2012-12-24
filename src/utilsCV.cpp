/*
 * utilsCV.cpp
 *
 *  Created on: 18/12/2012
 *      Author: xescriche
 */
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

bool readImagesFromFile(const string& imagesFilename,vector <Mat>& imagesVector, vector<string>& imagesVectorNames, int color, int & numImagesTotal) {
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
		numImagesTotal = readImageCount;
//		cout << "Number of images from file:                        " << readImageCount << endl;
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
}

void computeDescriptorsImagesVector(const vector<Mat>& imagesVector, vector<vector<KeyPoint> >& imagesVectorKeypointsVector, vector<Mat>& imagesVectorDescriptors,Ptr<DescriptorExtractor>& descriptorExtractor/*, int & numImagesVectorDescriptors*/) {
        descriptorExtractor->compute(imagesVector, imagesVectorKeypointsVector, imagesVectorDescriptors);
}

void showKeypoints(const vector<Mat>& vocabularyImages, const vector<vector<KeyPoint> >& vocabularyImagesKeypoints) {
	for (size_t i = 0; i < vocabularyImages.size(); i++) {
		Mat img_keypoints;
		drawKeypoints(vocabularyImages[i], vocabularyImagesKeypoints[i], img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		stringstream ss;
		ss << "Keypoints image " << i + 1;
		imshow(ss.str(), img_keypoints);
		waitKey(0);
	}
}

void kmeansVocabularyImages(const vector<Mat>& imagesVectorDescriptors, int clusterCount, int attempts,int numImagesTotal, vector<vector<int> >& vocabulary) {
	// POINT 2: APPLY KMEANS TO THE vocabularyImagesKeypoints SET
	int numRowsTotal = 0;
	for (unsigned int i = 0; i < imagesVectorDescriptors.size(); i++) {
		numRowsTotal = numRowsTotal + imagesVectorDescriptors[i].rows;
	}

	Mat src = imagesVectorDescriptors[0];
	Mat samples(numRowsTotal, src.cols, src.type());
	int jj = 0; //sample, labels index
	for (int i = 0; i < numImagesTotal; i++) {
		src = imagesVectorDescriptors[i];
		for (int j = 0; j < src.rows; j++) {
			for (int x = 0; x < src.cols; x++) {
				samples.at<float>(jj, x) = src.at<float>(j, x);
			}
			jj++;
		}
	}
	Mat labels;
	Mat centers;
	kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), attempts, KMEANS_PP_CENTERS, centers);

	// Show kmeans results
	cout<<"labels:"<<endl;
	for(int i = 0; i < labels.rows; ++i)
		cout<< "i:" << i << " label: " << labels.at<int>(0, i) << endl;

	// First we put all values to 0
	for(int i = 0; i < clusterCount; ++i) {
		for (int j = 0; j < numImagesTotal; j++) {
			vocabulary[i][j]=0;
		}
	}

	cout<<"\ncenters:"<<endl;

	// i=> centers index
	for(int i = 0; i < clusterCount; ++i) {
		jj=0;
		cout<< "i: " << i << " centerValue: " << centers.at<float>(0, i) << endl;
		// x=> image index
		for (int x = 0; x < numImagesTotal; x++) {
			src = imagesVectorDescriptors[x];
			for (int j = 0; j < src.rows; j++) {
				if (labels.at<int>(0,jj)==i) {  // The image contains a k=i
					vocabulary[i][x]=1;
				}
				jj++;
			}
		}
	}

	// Show vocabulary matrix
	for(int i = 0; i < clusterCount ; ++i) {
		for (int j = 0; j < numImagesTotal; j++) {
			cout<< "Vocabulary i: " << i << " j: " << j << " value: " << vocabulary[i][j] <<endl;
		}
	}
}


