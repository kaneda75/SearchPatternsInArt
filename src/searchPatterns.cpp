#include "utils.hpp"
#include "utilsCV.hpp"

using namespace cv;
using namespace std;

// Directories, files
const string vocabularyImagesNameFile = "/Users/xescriche/git/SearchPatternsInArt/tests/test1/vocabularyImages.txt";
const string newImageFileName = "/Users/xescriche/git/SearchPatternsInArt/tests/test1/tapies1.jpg";
const string dirToSaveResImages = "/Users/xescriche/git/SearchPatternsInArt/tests/test1/results1";

void searchPatterns(string algorithmType, int hessianThresholdSURF, bool uprightSURF, int k, int kIncrement, int criteriaKMeans, int attemptsKMeans, int minimumPointsOnVotes, int thresholdDistanceAdmitted, int kernelSize, bool resizeImage) {
	int numImagesTotal = 0;
	try {

	// POINT 1: DEFINE Feature detector (detectorType) AND Descriptor extractor (descriptorType)

		Ptr<FeatureDetector> featureDetector = FeatureDetector::create(algorithmType);
		if (featureDetector.empty()) cout << "The detector cannot be created." << endl << ">" << endl;

		Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create(algorithmType);
		if (descriptorExtractor.empty()) cout << "The descriptor cannot be created." << endl << ">" << endl;

		if (algorithmType == "SURF")
			createSurfDetector(hessianThresholdSURF, uprightSURF,featureDetector);

	// POINT 2.1: READ IMAGES, DETECT KEYPOINTS AND EXTRACT DESCRIPTORS ON "VOCABULARY IMAGES"

		vector<Mat> vocabularyImages;
		vector<string> vocabularyImagesNames;
		if (!readImagesFromFile(vocabularyImagesNameFile, vocabularyImages,vocabularyImagesNames, numImagesTotal)) cout << endl;

		vector<vector<KeyPoint> > vocabularyImagesKeypoints;
		detectKeypointsImagesVector(vocabularyImages, vocabularyImagesKeypoints, featureDetector);

		vector<Mat> imagesVectorDescriptors;
		computeDescriptorsImagesVector(vocabularyImages, vocabularyImagesKeypoints,imagesVectorDescriptors, descriptorExtractor);

		int numRowsTotal = calculeNumRowsTotal(imagesVectorDescriptors);
		cout << "Number of descriptors in vocabulary set: " << numRowsTotal << endl;

	// POINT 2.2: READ IMAGE, APPLY EFFECTS, DETECT KEYPOINTS AND EXTRACT DESCRIPTORS ON "NEW IMAGE"

		Mat newImage;
		if (!readImage(newImageFileName,newImage,0)) cout << endl;

		if (kernelSize != -1)
			applyGaussianBlur(newImage, kernelSize); // Gaussian Blur Effect

		if (resizeImage)
			applyResizeEffect(newImage);  // Resize Image Effect

		vector<KeyPoint> newImageKeypoints;
		detectKeypointsImage(newImage, newImageKeypoints, featureDetector);

		Mat newImageDescriptors;
		computeDescriptorsImage(newImage, newImageKeypoints, newImageDescriptors, descriptorExtractor);

	// POINT 3: K-MEANS

		int clusterCount = k;
		while (clusterCount <= numRowsTotal) {

		// POINT 3.1: Apply KMeans on vocabulary (imagesVectorDescriptors)

			vector<vector<int> > vocabulary(clusterCount, vector<int>(numImagesTotal));
			Mat centers;
			cout << endl;
			cout << "k: " << clusterCount << endl;
			kmeansVocabularyImages(imagesVectorDescriptors, clusterCount, criteriaKMeans, attemptsKMeans, numImagesTotal, vocabulary, centers, numRowsTotal);

		//  POINT 3.2: Find the KCenters on newImageDescriptors

			Mat kcentersNewImage(newImageDescriptors.rows, 1, centers.type());
			findKCentersOnNewImage(kcentersNewImage, newImageDescriptors, centers);

		// POINT 3.3: Voting images (Construct a Mat "matVote" with the labels/pattern that contains every image on vocabulary)

			Mat matVote = votingImages(vocabulary,kcentersNewImage,numImagesTotal);
			cout << endl;

	// POINT 4: RANSAC

			// POINT 4.1: Create a new imageResult. TODO: Create a copy of newImage without reading file again

			Mat imageResult;
			readImage(newImageFileName,imageResult,1); // read the image in color to put the result on GREEN and RED

			// POINT 4.2: For every voted image on vocabulary

			for (int imag = 0; imag < matVote.rows; ++imag) {
				if (matVote.at<int>(imag,0) >= minimumPointsOnVotes) {
					cout << "Image selected: " << imag << " with " << matVote.at<int>(imag,0) << " votes." << endl;

					// POINT 4.3: Select the image and find the KCenters

					Mat imageSelected = vocabularyImages[imag];
					vector<KeyPoint> imageSelectedKeypoints = vocabularyImagesKeypoints[imag];
					Mat imageSelectedDescriptors = imagesVectorDescriptors[imag];
					Mat kcentersImageSelected(imageSelectedDescriptors.rows, 1, centers.type());
					findKCentersOnNewImage(kcentersImageSelected, imageSelectedDescriptors, centers);

					// POINT 4.4: Apply RANSAC

					ransac(kcentersImageSelected, kcentersNewImage, imageSelected, imageSelectedKeypoints, newImage, newImageKeypoints, clusterCount, dirToSaveResImages, imag, thresholdDistanceAdmitted, imageResult);
					cout << endl;
				}
			}
			clusterCount = clusterCount + kIncrement;
		}
	} catch (exception& e) {
		cout << e.what() << endl;
	}
}

int main(int argc, char *argv[]) {

	// detector, descriptor types
	string algorithmType = "SIFT";		// Detector and descriptor type (Ex: "SURF", "SIFT")
	bool uprightSURF = false;			// (Only for SURF). This is USURF. false=detector computes orientation of each feature. true= the orientation is not computed.
	int hessianThresholdSURF = 500;		// (Only for SURF). Threshold for the keypoint detector. A good default value could be from 300 to 500, depending from the image contrast.

	// K-Means
	int initialK = 1; 					// Initial K Center constant in k-means. This must be <= Total number of rows in the sum of all vocabulary images.
	int kIncrement = 20;				// This is the increment of the k centers in kmeans loop
	int criteriaKMeans = 200;			// This is the maximum number of iterations in kmeans to recalcule the k-centers (Ex: 100 it's ok)
	int attemptsKMeans = 3;				// This is the number of times the algorithm is executed using different initial labellings (Ex: 3 it's ok)

	// RANSAC
	int minimumPointsOnVotes = 10;    	// Minimum number of votes that must to have every image to be selected. (Minimum 2.Homography needs 2 points minimum) (Ex: 8-10 are good values)
	int thresholdDistanceAdmitted = 10;	// Threshold distance admitted comparing distance between images on homography results.  (Ex: 3 it's ok)

	// Gaussian Blur
	int kernelSize = -1;				// This means the Gaussian kernel size applied to newImage. (-1: Not apply)
	bool resizeImage = false;			// This means if we make a resize transformation of the image

	searchPatterns(algorithmType, hessianThresholdSURF, uprightSURF, initialK, kIncrement, criteriaKMeans, attemptsKMeans, minimumPointsOnVotes,thresholdDistanceAdmitted, kernelSize, resizeImage);
}
