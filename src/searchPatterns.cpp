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
const int clusterCount = 10;  // K const in k-means. This must be <= Total number of rows in the sum of all vocabulary images.
const int attempts = 3;

// Directories, files
const string vocabularyImagesNameFile = "/Users/xescriche/git/SearchPatternsInArt/tests/test1/vocabularyImages.txt";
const string newImageFileName = "/Users/xescriche/git/SearchPatternsInArt/tests/test1/tapies3.jpg";
const string dirToSaveResImages = "/Users/xescriche/git/SearchPatternsInArt/tests/test1/results";

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

	//  POINT 3.2: Find KCenters on newImageDescriptors

		Mat matCenters(newImageDescriptors.rows, 1, centers.type());
		findKCentersOnNewImage(matCenters, newImageDescriptors, clusterCount, attempts, labels, centers);

	// POINT 3.3: Voting
		votingImages(vocabulary,matCenters,numImagesTotal);


	} catch (exception& e) {
		cout << e.what() << endl;
	}
}

int main(int argc, char *argv[]) {
	computeMatching();
}
