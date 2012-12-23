#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core.hpp"
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

const string vocabularyImagesNameFile = "/Users/xescriche/git/SearchPatternsInArt/tests/test1/vocabularyImages.txt";
const string newImageFileName = "/Users/xescriche/git/SearchPatternsInArt/tests/test1/tapies3.jpg";
const string detectorType = "SIFT";
const string descriptorType = "SIFT";
const int color = 0;
const string dirToSaveResImages = "/Users/xescriche/git/SearchPatternsInArt/tests/test1/results";

void computeMatching() {
	try {

	// POINT 1: DETECT KEYPOINTS AND DESCRIPTORS OF VOCABULARY IMAGES (USING SIFT/SURF)

		// SIFT Feature detector
		Ptr<FeatureDetector> featureDetector;
		featureDetector = FeatureDetector::create(detectorType);
		if (featureDetector.empty())
			cout << "The detector cannot be created." << endl << ">" << endl;

		// SIFT Descriptor extractor
		Ptr<DescriptorExtractor> descriptorExtractor;
		descriptorExtractor = DescriptorExtractor::create(descriptorType);
		if (featureDetector.empty())
			cout << "The descriptor cannot be created." << endl << ">" << endl;

		// Vocabulary images
		vector<Mat> vocabularyImages;
		vector<string> vocabularyImagesNames;
		if (!readImagesFromFile(vocabularyImagesNameFile, vocabularyImages,vocabularyImagesNames, color)) {
			cout << endl;
		}

		vector<vector<KeyPoint> > vocabularyImagesKeypoints;
		detectKeypointsImagesVector(vocabularyImages, vocabularyImagesKeypoints, featureDetector);

		// Show the keypoints on screen
//		showKeypoints(vocabularyImages, vocabularyImagesKeypoints);

	// POINT 2: APPLY KMEANS TO THE vocabularyImagesKeypoints SET

		vector<Mat> imagesVectorDescriptors;
		computeDescriptorsImagesVector(vocabularyImages, vocabularyImagesKeypoints,imagesVectorDescriptors, descriptorExtractor);

		Mat src = imagesVectorDescriptors[0];
		if (src!=NULL) {
			Mat samples(imagesVectorDescriptors.size() * src.rows, src.cols, src.type());
			for ( unsigned int i = 0; i < imagesVectorDescriptors.size(); i++) {
				src = imagesVectorDescriptors[i];
				for (int j = 0; j < src.rows; j++)
					for( int x = 0; x < src.cols; x++ )
						samples.at<float>(i+j, x) = src.at<float>(j,x);
			}

			imshow( "clustered image before kmeans", samples );
			waitKey( 0 );
			int clusterCount = 10;
			Mat labels;
			int attempts = 5;
			Mat centers;
			kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), attempts, KMEANS_PP_CENTERS, centers );
			//		Mat new_image( samples.size(), samples.type() );
			//		for( int y = 0; y < samples.rows; y++ )
			//			for( int x = 0; x < samples.cols; x++ )
			//			{
			//				int cluster_idx = labels.at<int>(y + x*samples.rows,0);
			//				new_image.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
			//				new_image.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
			//				new_image.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
			//			}
			imshow( "clustered image after kmeans", samples );
			waitKey( 0 );
		} else {
			cout << "The imageVectorDescriptors is NULL." << endl;
		}


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
