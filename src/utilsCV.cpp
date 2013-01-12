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

bool readImagesFromFile(const string& imagesFilename,vector <Mat>& imagesVector, vector<string>& imagesVectorNames, int & numImagesTotal) {
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

void applyGaussianBlur(Mat newImage, int kernelSize) {
	Mat src, dst;
	src = newImage;
	GaussianBlur(src, dst, Size(kernelSize, kernelSize), 0, 0);
	newImage = dst;
}

void applyResizeEffect(Mat newImage) {
	Mat src, dst;
	src = newImage;
	resize(src, dst, Size(src.cols * 2, src.rows * 2));
	src = dst;
	resize(src, dst, Size(src.cols / 2, src.rows / 2));
	newImage = dst;
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

void createSurfDetector(int hessianThresholdSURF, bool uprightSURF, Ptr<FeatureDetector> featureDetector) {
	Ptr<Feature2D> surf = Algorithm::create<Feature2D>("Feature2D.SURF");
	if (surf.empty())
		CV_Error(CV_StsNotImplemented, "OpenCV was built without SURF support");

	surf->set("hessianThreshold", hessianThresholdSURF);
	surf->set("nOctaves", 4);
	surf->set("nOctaveLayers", 2);
	surf->set("upright", uprightSURF);
	surf->set("extended", true);
	featureDetector = surf;
}

int calculeNumRowsTotal(const vector<Mat>& imagesVectorDescriptors) {
	int numRowsTotal = 0;
	for (unsigned int i = 0; i < imagesVectorDescriptors.size(); i++)
		numRowsTotal = numRowsTotal + imagesVectorDescriptors[i].rows;
	return numRowsTotal;
}




// 2. KMEANS

Mat extractSamplesMap(const vector<Mat>& imagesVectorDescriptors, int numRowsTotal, int numImagesTotal) {
	Mat src = imagesVectorDescriptors[0];
	Mat samples(numRowsTotal, src.cols, src.type());
	int jj = 0;
	for (int i = 0; i < numImagesTotal; i++) {
		src = imagesVectorDescriptors[i];
		for (int j = 0; j < src.rows; j++) {
			for (int x = 0; x < src.cols; x++)
				samples.at<float>(jj, x) = src.at<float>(j, x);
			jj++;
		}
	}
	return samples;
}

void extractVocabulary(int clusterCount, int numImagesTotal, Mat src, const vector<Mat>& imagesVectorDescriptors, Mat labels, vector<vector<int> >& vocabulary) {
	for (int i = 0; i < clusterCount; ++i)
		for (int j = 0; j < numImagesTotal; j++)
			vocabulary[i][j] = 0; // First we put all values to 0

	for (int i = 0; i < clusterCount; ++i) {
		int jj = 0;
		for (int x = 0; x < numImagesTotal; x++) {
			src = imagesVectorDescriptors[x];
			for (int j = 0; j < src.rows; j++) {
				if (labels.at<int>(0, jj) == i)
					vocabulary[i][x] = 1;
				jj++;
			}
		}
	}
}

void kmeansVocabularyImages(const vector<Mat>& imagesVectorDescriptors, int clusterCount,int criteriaKMeans, int attemptsKMeans,int numImagesTotal, vector<vector<int> >& vocabulary, Mat& centers, int numRowsTotal) {
	Mat labels;
	Mat src = imagesVectorDescriptors[0];
	Mat samples = extractSamplesMap(imagesVectorDescriptors, numRowsTotal, numImagesTotal);
	kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER, criteriaKMeans, 1.0), attemptsKMeans, KMEANS_PP_CENTERS, centers);
	extractVocabulary(clusterCount, numImagesTotal, src, imagesVectorDescriptors, labels, vocabulary);
}

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
	}
}

Mat votingImages(vector<vector<int> >& vocabulary,Mat& matCenters, int numImagesTotal) {
	Mat matVote(numImagesTotal, 1, matCenters.type());
	for (int i = 0; i < matVote.rows; ++i)
		matVote.at<int>(i,0) = 0; 				 // initialize matVote to 0

	for (int i = 0; i < matCenters.rows; ++i) {  // For each matCenters row
		int k = matCenters.at<int>(i,0);  		 // Get the kCenter value
		for (int j = 0; j < numImagesTotal; ++j)
			if (vocabulary[k][j] == 1)
				matVote.at<int>(j,0) = matVote.at<int>(j,0) + 1;
	}
	return matVote;
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


void getPointsVectors(const Mat& wordsImageIni, const Mat& wordsNewImage,const vector<KeyPoint>& imageIniKeypoints,const vector<KeyPoint>& newImageKeypoints, vector<Point2f>& obj, vector<Point2f>& scene, vector <pair <int, int> >& aMatches) {
	int wordIni;
	for (int i = 0; i < wordsImageIni.rows; ++i) {
		wordIni = wordsImageIni.at<int>(i, 0);
		for (int j = 0; j < wordsNewImage.rows; ++j)
			if (wordIni == wordsNewImage.at<int>(j, 0)) {
				obj.push_back(imageIniKeypoints[i].pt);
				scene.push_back(newImageKeypoints[j].pt);
				aMatches.push_back(make_pair(i,j));
			}
	}
}

bool isGoodHomography(const vector<Point2f>& sceneCorners, int thresholdDistanceAdmitted) {
	bool goodHomography = true;

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

void ransac(const Mat& wordsImageIni,const Mat& wordsNewImage, Mat imageIni,const vector<KeyPoint>& imageIniKeypoints, Mat newImage,const vector<KeyPoint>& newImageKeypoints, int clusterCount,const string dirToSaveResImages, int imag, int thresholdDistanceAdmitted, Mat imageResult) {
	vector<Point2f> obj, scene;
	vector <pair <int, int> > aMatches;
	getPointsVectors(wordsImageIni, wordsNewImage, imageIniKeypoints, newImageKeypoints, obj, scene,aMatches);
	cout << "# Descriptors: " << imageIniKeypoints.size() << endl;
	cout << "# Matches: " << obj.size() << endl;

	if (obj.size() >= 4) {  // findHomography needs minimum 4 points
		Mat imageResult2 = createMatchers(imageIni,newImage,imageIniKeypoints,newImageKeypoints,aMatches);
		Mat transform = findHomography(obj, scene, CV_RANSAC);
		double det = determinant(transform);
		cout << "# Determinant: " << det << endl;

		vector<Point2f> objCorners = getCorners(imageIni);
		vector<Point2f> sceneCorners(4);
		perspectiveTransform(objCorners, sceneCorners, transform);

		bool goodHomography = isGoodHomography(sceneCorners, thresholdDistanceAdmitted);
		if (goodHomography) {
			cout << "GOOD HOMOGRAPHY " << endl;
			drawImageLines(sceneCorners , imageIni, imageResult2, 1);
			drawImageLinesOnlyResultImage(sceneCorners , imageIni, imageResult, 1);
//		} else {
//			This part paints bad homographies in red. For the moment it's not relevant
//			drawImageLines(sceneCorners , imageIni, imageResult2, 0);
//			drawImageLinesOnlyResultImage(sceneCorners , imageIni, imageResult, 0);
		}
		saveImageResult(dirToSaveResImages, clusterCount, imag, imageResult2);
		saveImageResult2(dirToSaveResImages, clusterCount, imageResult);
	}
}
