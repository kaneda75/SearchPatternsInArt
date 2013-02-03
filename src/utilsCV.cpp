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

// 1. IMAGES

bool readImage(const string& imageName, Mat& image, int color) {
	image = imread(imageName, color);
	if (image.empty()) {
		cout << "The image can not be read." << endl << ">" << endl;
		return false;
	}
	return true;
}

bool readImagesFromFile(const string& imagesFilename,vector <Mat>& imagesVector, vector<string>& imagesVectorNames, int & numImagesTotal, int kernelSize) {
	string trainDirName;
	readVocabularyImages(imagesFilename, trainDirName, imagesVectorNames);
	if (imagesVectorNames.empty()) {
		cout << "The images cannot be read." << endl << ">" << endl;
		return false;
	}
	int readImageCount = 0;
	for (size_t i = 0; i < imagesVectorNames.size(); i++) {
		string filename = trainDirName + imagesVectorNames[i];
		Mat img = imread(filename, 0);
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
	}
    return true;
}

void applyGaussianBlur(Mat queryImage, int kernelSize) {
	GaussianBlur(queryImage, queryImage, Size(kernelSize, kernelSize), 0);
}

void applyResizeEffect(Mat& queryImage) {
	resize(queryImage, queryImage, Size(queryImage.cols/2, queryImage.rows/2), 0.5,0.5, INTER_LINEAR);
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

void createSurfDetector(int hessianThresholdSURF, int nOctaves, int nOctaveLayers, bool extended, bool uprightSURF, Ptr<FeatureDetector> featureDetector) {
	Ptr<Feature2D> surf = Algorithm::create<Feature2D>("Feature2D.SURF");
	if (surf.empty())
		CV_Error(CV_StsNotImplemented, "OpenCV was built without SURF support");
	surf->set("hessianThreshold", hessianThresholdSURF);
	surf->set("nOctaves", nOctaves);
	surf->set("nOctaveLayers", nOctaveLayers);
	surf->set("upright", uprightSURF);
	surf->set("extended", extended);
	featureDetector = surf;
}

int calculeNumRowsTotal(const vector<Mat>& imagesVectorDescriptors) {
	int numRowsTotal = 0;
	for (unsigned int i = 0; i < imagesVectorDescriptors.size(); i++)
		numRowsTotal = numRowsTotal + imagesVectorDescriptors[i].rows;
	return numRowsTotal;
}


// 2. KMEANS

Mat extractAllVocabularyDescriptors(const vector<Mat>& imagesVectorDescriptors, int numRowsTotal, int numImagesTotal) {
	Mat src = imagesVectorDescriptors[0];
	Mat vocabularyDescriptors(numRowsTotal, src.cols, src.type());
	int jj = 0;
	for (int i = 0; i < numImagesTotal; i++) {
		src = imagesVectorDescriptors[i];
		for (int j = 0; j < src.rows; j++) {
			for (int x = 0; x < src.cols; x++)
				vocabularyDescriptors.at<float>(jj, x) = src.at<float>(j, x);
			jj++;
		}
	}
	return vocabularyDescriptors;
}

void extractLabelsVocabularyStructure(int clusterCount, int numImagesTotal, const vector<Mat>& imagesVectorDescriptors, Mat labels, vector<vector<int> >& labelsVocabularyStructure) {
	for (int i = 0; i < clusterCount; ++i)
		for (int j = 0; j < numImagesTotal; j++)
			labelsVocabularyStructure[i][j] = 0; // First we put all values to 0

	for (int i = 0; i < clusterCount; ++i) {
		int jj = 0;
		for (int x = 0; x < numImagesTotal; x++) {
			Mat src = imagesVectorDescriptors[x];
			for (int j = 0; j < src.rows; j++) {
				if (labels.at<int>(0, jj) == i)
					labelsVocabularyStructure[i][x] = 1;
				jj++;
			}
		}
	}
}

void kmeansVocabularyImages(const vector<Mat>& imagesVectorDescriptors, int clusterCount,int criteriaKMeans, int attemptsKMeans,int numImagesTotal, vector<vector<int> >& labelsVocabularyStructure, Mat& centers, int numRowsTotal) {
	Mat vocabularyDescriptors = extractAllVocabularyDescriptors(imagesVectorDescriptors, numRowsTotal, numImagesTotal);
	Mat labels;
	kmeans(vocabularyDescriptors, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER, criteriaKMeans, 1.0), attemptsKMeans, KMEANS_PP_CENTERS, centers);
	extractLabelsVocabularyStructure(clusterCount, numImagesTotal, imagesVectorDescriptors, labels, labelsVocabularyStructure);
}

void findKCentersOnImage(Mat& matKCenters, Mat& imageDescriptors, Mat& centers) {
	for (int var = 0; var < imageDescriptors.rows; ++var) {  //For each descriptor's new image
		Mat descriptor = imageDescriptors.row(var);
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
		matKCenters.at<int>(var, 0) = position;
	}
}

Mat votingImages(vector<vector<int> >& labelsVocabularyStructure,Mat& kcentersQueryImage, int numImagesTotal) {
	Mat matVotes(numImagesTotal, 1, kcentersQueryImage.type());
	for (int i = 0; i < matVotes.rows; ++i)
		matVotes.at<int>(i,0) = 0; 				 // initialize matVote to 0

	for (int i = 0; i < kcentersQueryImage.rows; ++i) {
		int k = kcentersQueryImage.at<int>(i,0);  		 // Get the kCenter value
		for (int j = 0; j < numImagesTotal; ++j)
			if (labelsVocabularyStructure[k][j] == 1)
				matVotes.at<int>(j,0) = matVotes.at<int>(j,0) + 1;
	}
	return matVotes;
}


// 3. RANSAC

// Get the corners from the imageSelected ( the object to be "detected" )
vector<Point2f> getCorners(const Mat& imageSelected) {
	vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0);
	obj_corners[1] = cvPoint(imageSelected.cols, 0);
	obj_corners[2] = cvPoint(imageSelected.cols, imageSelected.rows);
	obj_corners[3] = cvPoint(0, imageSelected.rows);
	return obj_corners;
}

// Draw lines between the corners (two images: selected and result )
void drawImageLines(const vector<Point2f>& scene_corners,Mat imageSelected, Mat& imgResult, int color) {
	if (color == 1) { // good result. Green color.
		line(imgResult, scene_corners[0] + Point2f(imageSelected.cols, 0),scene_corners[1] + Point2f(imageSelected.cols, 0),Scalar(0, 255, 0), 4);
		line(imgResult, scene_corners[1] + Point2f(imageSelected.cols, 0),scene_corners[2] + Point2f(imageSelected.cols, 0),Scalar(0, 255, 0), 4);
		line(imgResult, scene_corners[2] + Point2f(imageSelected.cols, 0),scene_corners[3] + Point2f(imageSelected.cols, 0),Scalar(0, 255, 0), 4);
		line(imgResult, scene_corners[3] + Point2f(imageSelected.cols, 0),scene_corners[0] + Point2f(imageSelected.cols, 0),Scalar(0, 255, 0), 4);
	}
	if (color == 0) { // bad result. Red color
		line(imgResult, scene_corners[0] + Point2f(imageSelected.cols, 0),scene_corners[1] + Point2f(imageSelected.cols, 0),Scalar(0, 0, 255), 4);
		line(imgResult, scene_corners[1] + Point2f(imageSelected.cols, 0),scene_corners[2] + Point2f(imageSelected.cols, 0),Scalar(0, 0, 255), 4);
		line(imgResult, scene_corners[2] + Point2f(imageSelected.cols, 0),scene_corners[3] + Point2f(imageSelected.cols, 0),Scalar(0, 0, 255), 4);
		line(imgResult, scene_corners[3] + Point2f(imageSelected.cols, 0),scene_corners[0] + Point2f(imageSelected.cols, 0),Scalar(0, 0, 255), 4);
	}
}

// Draw lines between the corners (only result image)
void drawImageLinesOnlyResultImage(const vector<Point2f>& scene_corners,Mat imageSelected, Mat& imgResult, int color) {
	if (color == 1) { // good result. Green color.
		line(imgResult, scene_corners[0],scene_corners[1],Scalar(0, 255, 0), 4);
		line(imgResult, scene_corners[1],scene_corners[2],Scalar(0, 255, 0), 4);
		line(imgResult, scene_corners[2],scene_corners[3],Scalar(0, 255, 0), 4);
		line(imgResult, scene_corners[3],scene_corners[0],Scalar(0, 255, 0), 4);
	}
	if (color == 0) { // bad result. Red color
		line(imgResult, scene_corners[0] + Point2f(imageSelected.cols, 0),scene_corners[1] + Point2f(imageSelected.cols, 0),Scalar(0, 0, 255), 4);
		line(imgResult, scene_corners[1] + Point2f(imageSelected.cols, 0),scene_corners[2] + Point2f(imageSelected.cols, 0),Scalar(0, 0, 255), 4);
		line(imgResult, scene_corners[2] + Point2f(imageSelected.cols, 0),scene_corners[3] + Point2f(imageSelected.cols, 0),Scalar(0, 0, 255), 4);
		line(imgResult, scene_corners[3] + Point2f(imageSelected.cols, 0),scene_corners[0] + Point2f(imageSelected.cols, 0),Scalar(0, 0, 255), 4);
	}
}

void saveImageResult(const string& dirToSaveResImages, int clusterCount, int imag, const Mat& imageResult) {
	stringstream ss;
	ss << dirToSaveResImages << "/k_" << clusterCount << "-" << imag << ".jpg";
	string filename = ss.str();
	if (!imwrite(filename, imageResult)) cout << "The file " << filename << " cannot be saved in " << dirToSaveResImages << "." << endl;
}

void saveImageResult2(const string& dirToSaveResImages, int clusterCount, const Mat& imageResult) {
	stringstream ss;
	ss << dirToSaveResImages << "/total_k_" << clusterCount  << ".jpg";
	string filename = ss.str();
	if (!imwrite(filename, imageResult)) cout << "The file " << filename << " cannot be saved in " << dirToSaveResImages << "." << endl;
}


void getPointsVectors(const Mat& kcentersImageSelected, const Mat& kcentersQueryImage,const vector<KeyPoint>& imageSelectedKeypoints,const vector<KeyPoint>& queryImageKeypoints, vector<Point2f>& obj, vector<Point2f>& scene, vector <pair <int, int> >& aMatches) {
	int wordIni;
	for (int i = 0; i < kcentersImageSelected.rows; ++i) {
		wordIni = kcentersImageSelected.at<int>(i, 0);
		for (int j = 0; j < kcentersQueryImage.rows; ++j)
			if (wordIni == kcentersQueryImage.at<int>(j, 0)) {
				obj.push_back(imageSelectedKeypoints[i].pt);
				scene.push_back(queryImageKeypoints[j].pt);
				aMatches.push_back(make_pair(i,j));
			}
	}
}

bool isGoodHomography(const vector<Point2f>& sceneCorners, int thresholdDistanceAdmitted, double det) {
	bool goodHomography = true;
//	if (RelDif(det, 1) <= 0.1)
//			goodHomography = true;
//		else
//			goodHomography = false;
	if (abs(sceneCorners[0].x - sceneCorners[3].x) > thresholdDistanceAdmitted)		// Condition 1
		goodHomography = false;
	else
		if (abs(sceneCorners[0].y - sceneCorners[1].y) > thresholdDistanceAdmitted) 	// Condition 2
			goodHomography = false;
		else
			if (abs(sceneCorners[1].x - sceneCorners[2].x) > thresholdDistanceAdmitted) 	// Condition 3
				goodHomography = false;
			else
				if (abs(sceneCorners[2].y - sceneCorners[3].y) > thresholdDistanceAdmitted) 	// Condition 4
					goodHomography = false;
				else
					if (sceneCorners[0].x >= sceneCorners[1].x || sceneCorners[0].x >= sceneCorners[2].x) 	// Condition 5
						goodHomography = false;
					else
						if (sceneCorners[3].x >= sceneCorners[1].x || sceneCorners[3].x >= sceneCorners[2].x) 	// Condition 6
							goodHomography = false;
						else
							if (sceneCorners[0].y >= sceneCorners[2].y || sceneCorners[0].y >= sceneCorners[3].y) 	// Condition 7
								goodHomography = false;
							else
								if (sceneCorners[1].y >= sceneCorners[2].y || sceneCorners[1].y >= sceneCorners[3].y) 	// Condition 8
									goodHomography = false;
	return goodHomography;
}

Mat createMatchers(Mat& pic1, Mat& pic2, const std::vector <KeyPoint> &feats1, const std::vector <KeyPoint> &feats2, vector <pair <int, int> > aMatches) {
	vector <DMatch> matches;
	Mat output;
	matches.reserve((int)aMatches.size());
	for (int i=0; i < (int)aMatches.size(); ++i)
	    matches.push_back(DMatch(aMatches[i].first, aMatches[i].second, numeric_limits<float>::max()));
	drawMatches(pic1, feats1, pic2, feats2, matches, output);
	return output;
}

void removeInliers(vector<Point2f>& obj, vector<Point2f>& scene, const Mat& homography, vector<Point2f> sceneCorners,vector<Point2f>& obj2,vector<Point2f>& scene2) {
	Point2f pXY0 = sceneCorners[0];
	Point2f pXY1 = sceneCorners[1];
	Point2f pXY2 = sceneCorners[2];
	Point2f pXY3 = sceneCorners[3];
	float Xmin,Xmax,Ymin,Ymax;

	if (pXY0.x > pXY3.x) Xmin = pXY3.x; else Xmin = pXY0.x;
	if (pXY0.y > pXY1.y) Ymin = pXY1.y; else Ymin = pXY0.y;
	if (pXY1.x > pXY2.x) Xmax = pXY1.x; else Xmax = pXY2.x;
	if (pXY2.y > pXY3.y) Ymax = pXY2.y; else Ymax = pXY3.y;

	for (unsigned int i = 0; i < obj.size(); ++i) {
		if (scene.at(i).x >= Xmin && scene.at(i).x <= Xmax && scene.at(i).y >= Ymin && scene.at(i).y <= Ymax) {
		} else {
			obj2.push_back(obj.at(i));
			scene2.push_back(scene.at(i));
		}
	}
}

void computeHomography(vector<Point2f>& obj, vector<Point2f>& scene, const Mat& imageSelected, int thresholdDistanceAdmitted, int imag, Mat& imageResult, Mat& imageResult2) {
	if (obj.size() >= 4) {  // findHomography needs minimum 4 points
		Mat homography = findHomography(obj, scene, CV_RANSAC);
		double det = determinant(homography);
		vector<Point2f> objCorners = getCorners(imageSelected);
		vector<Point2f> sceneCorners(4);
		perspectiveTransform(objCorners, sceneCorners, homography);
		bool goodHomography = isGoodHomography(sceneCorners, thresholdDistanceAdmitted, det);
		if (goodHomography) {
			cout << "GOOD HOMOGRAPHY on Image: " << imag << endl;
			drawImageLinesOnlyResultImage(sceneCorners, imageSelected,imageResult, 1);
			drawImageLines(sceneCorners, imageSelected, imageResult2, 1);
			vector<Point2f> obj2, scene2;
			removeInliers(obj, scene, homography, sceneCorners, obj2, scene2);
			obj = obj2;
			scene = scene2;
		}
	}
}

void ransac(const Mat& kcentersImageSelected,const Mat& kcentersQueryImage, Mat imageSelected,const vector<KeyPoint>& imageSelectedKeypoints, Mat queryImage,const vector<KeyPoint>& queryImageKeypoints, int clusterCount,const string dirToSaveResImages, int imag, int thresholdDistanceAdmitted, Mat imageResult, int homographyAttempts) {
	vector<Point2f> obj, scene;
	vector <pair <int, int> > aMatches;
	getPointsVectors(kcentersImageSelected, kcentersQueryImage, imageSelectedKeypoints, queryImageKeypoints, obj, scene, aMatches);
	Mat imageResult2 = createMatchers(imageSelected,queryImage,imageSelectedKeypoints,queryImageKeypoints,aMatches);

	while (scene.size() >= 4 && homographyAttempts > 0) {
		computeHomography(obj, scene, imageSelected, thresholdDistanceAdmitted, imag, imageResult, imageResult2);
		homographyAttempts = homographyAttempts - 1;
	}
	saveImageResult2(dirToSaveResImages, clusterCount, imageResult);
	saveImageResult(dirToSaveResImages, clusterCount, imag, imageResult2);
}
