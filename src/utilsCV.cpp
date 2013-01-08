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

Mat extractSamplesMap(const vector<Mat>& imagesVectorDescriptors, int numRowsTotal, int numImagesTotal) {
	Mat src = imagesVectorDescriptors[0];
	Mat samples(numRowsTotal, src.cols, src.type());
	int jj = 0;
	for (int i = 0; i < numImagesTotal; i++) { // i IMAGE NUMBER
		src = imagesVectorDescriptors[i];
		for (int j = 0; j < src.rows; j++) { // j ROWS
			for (int x = 0; x < src.cols; x++) { // x COLS
				samples.at<float>(jj, x) = src.at<float>(j, x);
			}
			jj++;
		}
	}
	return samples;
}

Mat addDescriptorToSamplesMap(Mat samples, Mat descriptor) {
	Mat samples2(samples.rows+1, samples.cols, samples.type());
	int i = 0;
	for (i = 0; i < samples.rows; i++) {
		for (int j = 0; j < samples.cols; j++) {
			samples2.at<float>(i, j) = samples.at<float>(i, j);
		}
	}
	i++;
	for (int x = 0; x < descriptor.cols; x++) {
		samples2.at<float>(i, x) = descriptor.at<float>(0, x);
	}
	return samples2;
}

int calculeNumRowsTotal(const vector<Mat>& imagesVectorDescriptors) {
	int numRowsTotal = 0;
	for (unsigned int i = 0; i < imagesVectorDescriptors.size(); i++) {
		numRowsTotal = numRowsTotal + imagesVectorDescriptors[i].rows;
	}
	return numRowsTotal;
}

void showVocabulary(int clusterCount, int numImagesTotal,
		vector<vector<int> >& vocabulary) {
	// Show vocabulary matrix
	for (int i = 0; i < clusterCount; ++i) {
		for (int j = 0; j < numImagesTotal; j++) {
//			cout << "Vocabulary i: " << i << " j: " << j << " value: " << vocabulary[i][j] << endl;
		}
	}
}

void extractVocabulary(int clusterCount, int numImagesTotal, Mat src, const vector<Mat>& imagesVectorDescriptors, Mat labels, vector<vector<int> >& vocabulary) {
	// First we put all values to 0
	for (int i = 0; i < clusterCount; ++i) {
		for (int j = 0; j < numImagesTotal; j++) {
			vocabulary[i][j] = 0;
		}
	}
//	cout << "\n" << "Centers:" << endl;
	// i=> centers index
	for (int i = 0; i < clusterCount; ++i) {
		int jj = 0;
		// x=> image index
		for (int x = 0; x < numImagesTotal; x++) {
			src = imagesVectorDescriptors[x];
			for (int j = 0; j < src.rows; j++) {
				if (labels.at<int>(0, jj) == i) {
					// The image contains a k=i
					vocabulary[i][x] = 1;
				}
				jj++;
			}
		}
	}
//	showVocabulary(clusterCount, numImagesTotal, vocabulary);
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

// POINT 2: APPLY KMEANS ON imagesVectorDescriptors
void kmeansVocabularyImages(const vector<Mat>& imagesVectorDescriptors, int clusterCount,int criteriaKMeans, int attemptsKMeans,int numImagesTotal, vector<vector<int> >& vocabulary, Mat& labels, Mat& centers, int numRowsTotal) {
	Mat src = imagesVectorDescriptors[0];
	Mat samples = extractSamplesMap(imagesVectorDescriptors, numRowsTotal, numImagesTotal);
	kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER, criteriaKMeans, 1.0), attemptsKMeans, KMEANS_PP_CENTERS, centers);
//	showMatrixValues2(labels, "labels:");
//	showMatrixValues2(samples, "samples:");
//	showMatrixValues2(centers, "centers:");
	extractVocabulary(clusterCount, numImagesTotal, src, imagesVectorDescriptors, labels, vocabulary);
}


// POINT 3.2: Find KCenters on newImageDescriptors
void findKCentersOnNewImage(Mat& matCenters, Mat& newImageDescriptors, Mat& centers) {
	for (int var = 0; var < newImageDescriptors.rows; ++var) {  //For each descriptor's new image
		Mat descriptor = newImageDescriptors.row(var);
		Mat matDifference(centers.rows, 1, centers.type());
		for (int i = 0; i < centers.rows; ++i) {  // For each k center
			float sumTotal = 0;
			Mat src = centers.row(i);
			for (int j = 0; j < src.cols; ++j) {  // Comparing values
				float diference = abs(src.at<float>(0, j) - descriptor.at<float>(0, j));
				sumTotal = sumTotal + diference;
			}
			matDifference.at<float>(i,0) = sumTotal;
		}

		float value = matDifference.at<float>(0,0);
		int position = 0;
		for (int x = 1; x < matDifference.rows; ++x) {
			if (matDifference.at<float>(x,0) < value) {
				value = matDifference.at<float>(x,0);
				position = x;
			}
		}
		matCenters.at<int>(var, 0) = position;
//		showMatrixValues2(descriptor, "descriptor:");
//		cout << "New Image Row: " << var << " KCenter: " << position << endl;
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
	//	cout << "Most voted image: " << mostVotedImage << " with " << mostVotedImageValue << " votes." << endl;
	return mostVotedImage;
}

Mat votingImages(vector<vector<int> >& vocabulary,Mat& matCenters, int numImagesTotal) {

	Mat matVote(numImagesTotal, 1, matCenters.type());
	// initialize matVote to 0
	for (int i = 0; i < matVote.rows; ++i) {
		matVote.at<int>(i,0)=0;
	}

	for (int i = 0; i < matCenters.rows; ++i) {  // For each matCenters row
		int k = matCenters.at<int>(i,0);  // Get the kCenter value
		for (int j = 0; j < numImagesTotal; ++j) {
			if (vocabulary[k][j] == 1) {  // If the vocabulary
				matVote.at<int>(j,0) = matVote.at<int>(j,0) + 1;
			}
		}
	}
//	showMatrixValues2(matVote, "matVote:");
	return matVote;
}

// Get the corners from the imageSelected ( the object to be "detected" )
vector<Point2f> getCorners(const Mat& imageSelected) {
	vector<Point2f> obj_corners(4);

	obj_corners[0] = cvPoint(0, 0);
	obj_corners[1] = cvPoint(imageSelected.cols, 0);
	obj_corners[2] = cvPoint(imageSelected.cols, imageSelected.rows);
	obj_corners[3] = cvPoint(0, imageSelected.rows);

	return obj_corners;
}

// Draw lines between the corners (the mapped object in the scene - image_2 )
void drawImageLines(const vector<Point2f>& scene_corners,Mat imageSelected, Mat& imgResult) {
	line(imgResult, scene_corners[0] + Point2f(imageSelected.cols, 0),scene_corners[1] + Point2f(imageSelected.cols, 0),Scalar(0, 255, 0), 4);
	line(imgResult, scene_corners[1] + Point2f(imageSelected.cols, 0),scene_corners[2] + Point2f(imageSelected.cols, 0),Scalar(0, 255, 0), 4);
	line(imgResult, scene_corners[2] + Point2f(imageSelected.cols, 0),scene_corners[3] + Point2f(imageSelected.cols, 0),Scalar(0, 255, 0), 4);
	line(imgResult, scene_corners[3] + Point2f(imageSelected.cols, 0),scene_corners[0] + Point2f(imageSelected.cols, 0),Scalar(0, 255, 0), 4);
}

void getPointsVectors(const Mat& wordsImageIni, const Mat& wordsNewImage,const vector<KeyPoint>& imageIniKeypoints,const vector<KeyPoint>& newImageKeypoints, vector<Point2f>& obj, vector<Point2f>& scene) {
	int wordIni;
	for (int i = 0; i < wordsImageIni.rows; ++i) {
		wordIni = wordsImageIni.at<int>(i, 0);
		for (int j = 0; j < wordsNewImage.rows; ++j) {
			if (wordIni == wordsNewImage.at<int>(j, 0)) {
				obj.push_back(imageIniKeypoints[i].pt);
				scene.push_back(newImageKeypoints[j].pt);
//				cout << "Word: " << wordIni << " I1(X,Y): (" << imageIniKeypoints[i].pt.x << "," << imageIniKeypoints[i].pt.y << ") I2(X,Y): (" << newImageKeypoints[j].pt.x << "," << newImageKeypoints[j].pt.y << ")" << endl;
			}
		}
	}
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

void saveImageResult(const string& dirToSaveResImages, int clusterCount,
		int imag, const Mat& imageResult) {
	stringstream ss;
	ss << dirToSaveResImages << "/k_" << clusterCount << "-" << imag << ".jpg";
	string filename = ss.str();
	if (!imwrite(filename, imageResult))
		cout << "The file " << filename << " cannot be saved in "
				<< dirToSaveResImages << "." << endl;
}

bool isGoodHomography(const vector<Point2f>& sceneCorners,
		int thresholdDistanceAdmitted) {
	bool goodHomography = true;
	float diference = 0;
	// Condition 1
	diference = abs(sceneCorners[0].x - sceneCorners[3].x);
	if (diference > thresholdDistanceAdmitted) {
		goodHomography = false;
	} else {
		// Condition 2
		diference = abs(sceneCorners[0].y - sceneCorners[1].y);
		if (diference > thresholdDistanceAdmitted) {
			goodHomography = false;
		} else {
			// Condition 3
			diference = abs(sceneCorners[1].x - sceneCorners[2].x);
			if (diference > thresholdDistanceAdmitted) {
				goodHomography = false;
			} else {
				// Condition 4
				diference = abs(sceneCorners[2].y - sceneCorners[3].y);
				if (diference > thresholdDistanceAdmitted) {
					goodHomography = false;
				} else {
					// Condition 5
					if (sceneCorners[0].x >= sceneCorners[1].x
							|| sceneCorners[0].x >= sceneCorners[2].x) {
						goodHomography = false;
					} else {
						// Condition 6
						if (sceneCorners[3].x >= sceneCorners[1].x
								|| sceneCorners[3].x >= sceneCorners[2].x) {
							goodHomography = false;
						} else {
							// Condition 7
							if (sceneCorners[0].y >= sceneCorners[2].y
									|| sceneCorners[0].y >= sceneCorners[3].y) {
								goodHomography = false;
							} else {
								// Condition 8
								if (sceneCorners[1].y >= sceneCorners[2].y
										|| sceneCorners[1].y
												>= sceneCorners[3].y) {
									goodHomography = false;
								}
							}
						}
					}
				}
			}
		}
	}
	return goodHomography;
}

void ransac(const Mat& wordsImageIni,const Mat& wordsNewImage, Mat imageIni,const vector<KeyPoint>& imageIniKeypoints, Mat newImage,const vector<KeyPoint>& newImageKeypoints, int clusterCount,const string dirToSaveResImages, int imag, int thresholdDistanceAdmitted) {
	vector<Point2f> obj;
	vector<Point2f> scene;

	getPointsVectors(wordsImageIni, wordsNewImage, imageIniKeypoints, newImageKeypoints, obj, scene);

//	cout << "obj.size():" << obj.size() << endl;
//	cout << "scene.size():" << scene.size() << endl;

	Mat imageResult = createImageResult(imageIni, imageIniKeypoints, newImage, newImageKeypoints);

	// Find Homography
	Mat transform = findHomography(obj, scene, CV_RANSAC);
//	cout << "transform:" << endl;
//	dumpMatrix(transform);
//	cout << endl;

//	double det = determinant(transform);
//	cout << "determinant:" << det << endl;
//	cout << endl;
//
//	Mat w, u, vt;
//	SVD::compute(transform, w, u, vt);
//	cout << "w:" << endl;
//	dumpMatrix(w);
//	cout << endl;
//	cout << "u:" << endl;
//	dumpMatrix(u);
//	cout << endl;
//	cout << "vt:" << endl;
//	dumpMatrix(vt);
//	cout << endl;

	vector<Point2f> objCorners = getCorners(imageIni);
	vector<Point2f> sceneCorners(4);
	perspectiveTransform(objCorners, sceneCorners, transform);

//	cout << "objCorners: " << endl;
//	cout << objCorners << endl;
//	cout << "sceneCorners: " << endl;
//	cout << sceneCorners << endl;

	bool goodHomography = isGoodHomography(sceneCorners, thresholdDistanceAdmitted);
	if (goodHomography) {
		drawImageLines(sceneCorners , imageIni, imageResult);
		saveImageResult(dirToSaveResImages, clusterCount, imag, imageResult);
	}
}

