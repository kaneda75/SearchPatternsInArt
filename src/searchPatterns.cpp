#include "utils.hpp"
#include "utilsCV.hpp"

using namespace cv;
using namespace std;

// Detectors, descriptors, loadImage
const string detectorType = "SIFT";
const string descriptorType = "SIFT";
const int color = 0;
int numImagesTotal = 0;

// K-means
const int clusterCount = 20;  // K const in k-means. This must be <= Total number of rows in the sum of all vocabulary images.
const int attempts = 3;

// Directories, files
const string vocabularyImagesNameFile = "/Users/xescriche/git/SearchPatternsInArt/tests/test1/vocabularyImages.txt";
const string newImageFileName = "/Users/xescriche/git/SearchPatternsInArt/tests/test1/tapies1.jpg";
const string dirToSaveResImages = "/Users/xescriche/git/SearchPatternsInArt/tests/test1/results";

std::vector<DMatch> performeBestMatches(const Mat& imageSelectedDescriptors,
		const std::vector<DMatch>& matches) {
	double max_dist = 0;
	double min_dist = 100;
	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < imageSelectedDescriptors.rows; i++) {
		double dist = matches[i].distance;
		if (dist < min_dist)
			min_dist = dist;

		if (dist > max_dist)
			max_dist = dist;
	}
	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);
	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector<DMatch> good_matches;
	for (int i = 0; i < imageSelectedDescriptors.rows; i++) {
		if (matches[i].distance < 3 * min_dist) {
			good_matches.push_back(matches[i]);
		}
	}
	return good_matches;
}

std::vector<Point2f> getCorners(const Mat& imageSelected) {
	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0);
	obj_corners[1] = cvPoint(imageSelected.cols, 0);
	obj_corners[2] = cvPoint(imageSelected.cols, imageSelected.rows);
	obj_corners[3] = cvPoint(0, imageSelected.rows);
	return obj_corners;
}

void drawImageLines(const std::vector<Point2f>& scene_corners,Mat imageSelected, Mat& img_matches) {
	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line(img_matches, scene_corners[0] + Point2f(imageSelected.cols, 0),scene_corners[1] + Point2f(imageSelected.cols, 0),Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + Point2f(imageSelected.cols, 0),scene_corners[2] + Point2f(imageSelected.cols, 0),Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + Point2f(imageSelected.cols, 0),scene_corners[3] + Point2f(imageSelected.cols, 0),Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + Point2f(imageSelected.cols, 0),scene_corners[0] + Point2f(imageSelected.cols, 0),Scalar(0, 255, 0), 4);
}

void getKeypointsFromGoodMatches(const std::vector<DMatch>& good_matches,
		const vector<KeyPoint>& imageSelectedKeypoints,
		const vector<KeyPoint>& newImageKeypoints, std::vector<Point2f>& obj,
		std::vector<Point2f>& scene) {
	for (unsigned int i = 0; i < good_matches.size(); i++) {
		obj.push_back(imageSelectedKeypoints[good_matches[i].queryIdx].pt);
		scene.push_back(newImageKeypoints[good_matches[i].trainIdx].pt);
	}
}

void ransacEX1(const Mat& imageSelectedDescriptors,const Mat& newImageDescriptors, Mat imageSelected,const vector<KeyPoint>& imageSelectedKeypoints, const Mat& newImage,const vector<KeyPoint>& newImageKeypoints) {

	// Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector<DMatch> matches;
	matcher.match(imageSelectedDescriptors, newImageDescriptors, matches);
	std::vector<DMatch> good_matches = performeBestMatches(imageSelectedDescriptors, matches);

	Mat img_matches;
	drawMatches(imageSelected, imageSelectedKeypoints, newImage,newImageKeypoints, good_matches, img_matches, Scalar::all(-1),Scalar::all(-1), vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("Good Matches & Object detection", img_matches);
	waitKey(0);

	//Get the keypoints from the good matches
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	getKeypointsFromGoodMatches(good_matches, imageSelectedKeypoints, newImageKeypoints, obj, scene);

	// Find Homography
	Mat H = findHomography(obj, scene, CV_RANSAC);

	// Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners = getCorners(imageSelected);

	// Transform perspective
	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H);

	// Draw lines between the corners (the mapped object in the scene - image_2 )
	drawImageLines(scene_corners, imageSelected, img_matches);

	// Show detected matches
	imshow("Good Matches & Object detection", img_matches);
	waitKey(0);
}


void ransac(const Mat& wordsImageIni,const Mat& wordsNewImage, Mat imageIni,const vector<KeyPoint>& imageIniKeypoints, Mat newImage,const vector<KeyPoint>& newImageKeypoints) {
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	int wordIni;
	for (int i = 0; i < wordsImageIni.rows; ++i) {
		wordIni = wordsImageIni.at<int>(i,0);
		for (int j = 0; j < wordsNewImage.rows; ++j) {
			if (wordIni==wordsNewImage.at<int>(j,0)) {
				obj.push_back(imageIniKeypoints[i].pt);
				scene.push_back(newImageKeypoints[j].pt);
			}
		}
	}

	// Find Homography
	Mat H = findHomography(obj, scene, CV_RANSAC);

	// Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners = getCorners(imageIni);

	// Transform perspective
	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H);

	// Draw lines between the corners (the mapped object in the scene - image_2 )

	Mat imageResult;
	std::vector<DMatch> good_matches;
	drawMatches(imageIni, imageIniKeypoints, newImage,newImageKeypoints, good_matches, imageResult, Scalar::all(-1),Scalar::all(-1), vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);	drawImageLines(scene_corners, imageIni, imageResult);

	// Draw lines between the corners (the mapped object in the scene - image_2 )
	drawImageLines(scene_corners, newImage, imageResult);

	// Show detected matches
	imshow("Good Matches & Object detection", imageResult);
	waitKey(0);
}

void computeMatching() {
	try {

	// POINT 1: DEFINE Feature detector (detectorType) AND Descriptor extractor (descriptorType)

		// Feature detector
		Ptr<FeatureDetector> featureDetector;
		featureDetector = FeatureDetector::create(detectorType);
		if (featureDetector.empty())
			cout << "The detector cannot be created." << endl << ">" << endl;

		// Descriptor extractor
		Ptr<DescriptorExtractor> descriptorExtractor;
		descriptorExtractor = DescriptorExtractor::create(descriptorType);
		if (featureDetector.empty())
			cout << "The descriptor cannot be created." << endl << ">" << endl;

	// POINT 2.1: READ IMAGES, DETECT KEYPOINTS AND EXTRACT DESCRIPTORS ON "VOCABULARY IMAGES"

		vector<Mat> vocabularyImages;
		vector<string> vocabularyImagesNames;
		if (!readImagesFromFile(vocabularyImagesNameFile, vocabularyImages,vocabularyImagesNames, color, numImagesTotal))
			cout << endl;

		vector<vector<KeyPoint> > vocabularyImagesKeypoints;
		detectKeypointsImagesVector(vocabularyImages, vocabularyImagesKeypoints, featureDetector);

		// Show the keypoints on screen
		// showKeypoints(vocabularyImages, vocabularyImagesKeypoints);

		vector<Mat> imagesVectorDescriptors;
		computeDescriptorsImagesVector(vocabularyImages, vocabularyImagesKeypoints,imagesVectorDescriptors, descriptorExtractor);

	// POINT 2.2: KMEANS ON imagesVectorDescriptors

		vector<vector<int> > vocabulary(clusterCount, vector<int>(numImagesTotal));
		Mat labels;
		Mat centers;
		kmeansVocabularyImages(imagesVectorDescriptors, clusterCount, attempts, numImagesTotal, vocabulary, labels, centers);

    // POINT 3.1: READ IMAGE, DETECT KEYPOINTS AND EXTRACT DESCRIPTORS ON "NEW IMAGE"

		Mat newImage;
		if (!readImage(newImageFileName,newImage,color))
			cout << endl;

		vector<KeyPoint> newImageKeypoints;
		detectKeypointsImage(newImage, newImageKeypoints, featureDetector);

		// Show the keypoints on screen
		// showKeypointsImage(newImage, newImageKeypoints);

		Mat newImageDescriptors;
		computeDescriptorsImage(newImage, newImageKeypoints, newImageDescriptors, descriptorExtractor);

	//  POINT 3.2: Find Words/KCenters on newImageDescriptors

		Mat wordsNewImage(newImageDescriptors.rows, 1, centers.type());
		findKCentersOnNewImage(wordsNewImage, newImageDescriptors, clusterCount, attempts, labels, centers);

	// POINT 3.3: Voting images

		int mostVotedImage = votingImages(vocabulary,wordsNewImage,numImagesTotal);

	// POINT 4.1: RANSAC

//		ransacEX1(imageSelectedDescriptors, newImageDescriptors, imageSelected, imageSelectedKeypoints, newImage, newImageKeypoints);

	// Find Words/KCenters on imageSelectedDescriptors

		Mat imageSelected = vocabularyImages[mostVotedImage];
		vector<KeyPoint> imageSelectedKeypoints = vocabularyImagesKeypoints[mostVotedImage];
		Mat imageSelectedDescriptors = imagesVectorDescriptors[mostVotedImage];
		Mat wordsImageIni(imageSelectedDescriptors.rows, 1, centers.type());
		findKCentersOnNewImage(wordsImageIni, imageSelectedDescriptors, clusterCount, attempts, labels, centers);
		showMatrixValues3(imageSelectedKeypoints,wordsImageIni, "wordsImageIni:");
		showMatrixValues3(newImageKeypoints, wordsNewImage, "wordsNewImage:");
		ransac(wordsImageIni, wordsNewImage, imageSelected, imageSelectedKeypoints, newImage, newImageKeypoints);


   } catch (exception& e) {
		cout << e.what() << endl;
	}
}

int main(int argc, char *argv[]) {
	computeMatching();
}
