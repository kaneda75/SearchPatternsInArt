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
const int clusterCount = 4;  // K const in k-means. This must be <= Total number of rows in the sum of all vocabulary images.
const int attempts = 3;

// Directories, files
const string vocabularyImagesNameFile = "/Users/xescriche/git/SearchPatternsInArt/tests/test1/vocabularyImages.txt";
const string newImageFileName = "/Users/xescriche/git/SearchPatternsInArt/tests/test1/tapies3.jpg";
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
		if (!readImagesFromFile(vocabularyImagesNameFile, vocabularyImages,vocabularyImagesNames, color, numImagesTotal)) {
			cout << endl;
		}

		vector<vector<KeyPoint> > vocabularyImagesKeypoints;
		detectKeypointsImagesVector(vocabularyImages, vocabularyImagesKeypoints, featureDetector);

		// Show the keypoints on screen
//		showKeypoints(vocabularyImages, vocabularyImagesKeypoints);

		vector<Mat> imagesVectorDescriptors;
		computeDescriptorsImagesVector(vocabularyImages, vocabularyImagesKeypoints,imagesVectorDescriptors, descriptorExtractor);

	// POINT 2: APPLY KMEANS TO THE vocabularyImagesKeypoints SET

		vector<vector<int> > vocabulary(clusterCount, vector<int>(numImagesTotal));
		Mat samples = kmeansVocabularyImages(imagesVectorDescriptors, clusterCount, attempts, numImagesTotal, vocabulary);

    // POINT 3: NEW IMAGE

		Mat newImage;
		if (!readImage(newImageFileName,newImage,color)) {
			cout << endl;
		}

		// 3.1 SIFT on NEW IMAGE
		vector<KeyPoint> newImageKeypoints;
		detectKeypointsImage(newImage, newImageKeypoints, featureDetector);

		// Show the keypoints on screen
//		showKeypointsImage(newImage, newImageKeypoints);

		Mat newImageDescriptors;
		computeDescriptorsImage(newImage, newImageKeypoints, newImageDescriptors, descriptorExtractor);

		// 3.2 KMEANS on NEW IMAGE
//		kmeansNewImage(samples,newImageDescriptors, clusterCount, attempts);

	} catch (exception& e) {
		cout << e.what() << endl;
	}
}

int main(int argc, char *argv[]) {
	computeMatching();
}
