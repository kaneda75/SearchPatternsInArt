#include "utilsCV.hpp"
#include "backupUnusedFunctions.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;

void showKeypointsImagesVector(const vector<Mat>& vocabularyImages, const vector<vector<KeyPoint> >& vocabularyImagesKeypoints) {
	for (size_t i = 0; i < vocabularyImages.size(); i++) {
		Mat img_keypoints;
		drawKeypoints(vocabularyImages[i], vocabularyImagesKeypoints[i], img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		stringstream ss;
		ss << "Keypoints image " << i + 1;
		imshow(ss.str(), img_keypoints);
		waitKey(0);
	}
}

void showKeypointsImage(const Mat& image, const vector<KeyPoint> & imageKeypoints) {
		Mat img_keypoints;
		drawKeypoints(image, imageKeypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		stringstream ss;
		ss << "Keypoints new image";
		imshow(ss.str(), img_keypoints);
		waitKey(0);
}

void showVocabulary(int clusterCount, int numImagesTotal,
		vector<vector<int> >& vocabulary) {
	for (int i = 0; i < clusterCount; ++i) {
		for (int j = 0; j < numImagesTotal; j++) {
//			cout << "Vocabulary i: " << i << " j: " << j << " value: " << vocabulary[i][j] << endl;
		}
	}
}

void showMatrixValues(Mat& matrix, string s) {
	cout << s << endl;
	for (int i = 0; i < matrix.rows; ++i)
		cout << "i:" << i << " value: " << matrix.at<int>(0, i) << endl;
}

void showMatrixValues2(Mat& matrix, string s) {
	cout << endl << s << endl;
	for (int i = 0; i < matrix.rows; ++i) {
		stringstream ss;
		for (int j = 0; j < matrix.cols; ++j) {
			ss << matrix.at<int>(i, j) << ";";
		}
		cout << ss.str() << endl;
	}
}

void showMatrixValues3(vector<KeyPoint> keypoints, Mat& matrix,  string s) {
	cout << endl << "X         Y      Word" << s << endl;
	for (int i = 0; i < matrix.rows; ++i) {
		stringstream ss;
		for (int j = 0; j < matrix.cols; ++j) {
			ss << keypoints[i].pt.x << "   " << keypoints[i].pt.y << "      " << matrix.at<int>(i, j);
		}
		cout << ss.str() << endl;
	}
}

// Calcule the most voted image
int getMostVotedImage(Mat matVote) {
	int mostVotedImage = 0;
	int mostVotedImageValue = 0;
	for (int var = 0; var < matVote.rows; ++var) {
		if (matVote.at<int>(var, 0) > mostVotedImageValue) {
			mostVotedImage = var;
			mostVotedImageValue = matVote.at<int>(var, 0);
		}
	}
	return mostVotedImage;
}

Mat createImageResult(const Mat& imageIni, const vector<KeyPoint>& imageIniKeypoints, const Mat& newImage, const vector<KeyPoint>& newImageKeypoints) {
	Mat imageResult;
	vector<DMatch> matches;
	drawMatches(imageIni, imageIniKeypoints, newImage, newImageKeypoints, matches, imageResult, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	return imageResult;
}

void dumpMatrix(const Mat &mat) {
     const int t = mat.type();
     for (int i = 0; i < mat.rows; i++) {
         for (int j = 0; j < mat.cols; j++) {
             switch (t) {
             case CV_32F:
                 printf("%6.4f ", mat.at<float> (i, j));
                 break;
             case CV_64F:
                 printf("%6.4f ", mat.at<double> (i, j));
                 break;
             }
         }
         printf("\n");
     }
}
