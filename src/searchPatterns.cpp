#include "utils.hpp"
#include "utilsCV.hpp"

using namespace cv;
using namespace std;

// Directories, files
const string vocabularyImagesNameFile = "/Users/xescriche/git/SearchPatternsInArt/tests/exp2/test1-SIFT/vocabularyImages.txt";
const string queryImageFileName = "/Users/xescriche/git/SearchPatternsInArt/tests/exp2/test1-SIFT/tapies7.jpg";
const string dirToSaveResImages = "/Users/xescriche/git/SearchPatternsInArt/tests/exp2/test1-SIFT/results1";

void searchPatterns(string algorithmType, int hessianThresholdSURF, int nOctaves, int nOctaveLayers, bool extended,  bool uprightSURF, int k, int kIncrement, int criteriaKMeans, int attemptsKMeans, int minimumVotes, int thresholdDistanceAdmitted, int kernelSize, bool resizeImage, int homographyAttempts) {
	int numImagesTotal = 0;
	try {

	// POINT 1.1: DEFINE Feature detector (detectorType) AND Descriptor extractor (descriptorType)
		Ptr<FeatureDetector> featureDetector = FeatureDetector::create(algorithmType);
		if (featureDetector.empty()) cout << "The detector cannot be created." << endl << ">" << endl;

		Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create(algorithmType);
		if (descriptorExtractor.empty()) cout << "The descriptor cannot be created." << endl << ">" << endl;

		if (algorithmType == "SURF")
			createSurfDetector(hessianThresholdSURF, nOctaves, nOctaveLayers, extended, uprightSURF, featureDetector);

	// POINT 1.2: READ IMAGES, DETECT KEYPOINTS AND EXTRACT DESCRIPTORS ON "VOCABULARY IMAGES"
		vector<Mat> vocabularyImages;
		vector<string> vocabularyImagesNames;
		if (!readImagesFromFile(vocabularyImagesNameFile, vocabularyImages,vocabularyImagesNames, numImagesTotal, kernelSize)) cout << endl;

		vector<vector<KeyPoint> > vocabularyImagesKeypoints;
		detectKeypointsImagesVector(vocabularyImages, vocabularyImagesKeypoints, featureDetector);

		vector<Mat> imagesVectorDescriptors;
		computeDescriptorsImagesVector(vocabularyImages, vocabularyImagesKeypoints,imagesVectorDescriptors, descriptorExtractor);

		int numRowsTotal = calculeNumRowsTotal(imagesVectorDescriptors);
		cout << "Number of descriptors in vocabulary set: " << numRowsTotal << endl;

	// POINT 1.3: READ IMAGE, APPLY EFFECTS, DETECT KEYPOINTS AND EXTRACT DESCRIPTORS ON "QUERY IMAGE"
		Mat queryImage;
		if (!readImage(queryImageFileName,queryImage,0)) cout << endl;

		if (kernelSize != -1)
			applyGaussianBlur(queryImage ,kernelSize); // Gaussian Blur Effect

		if (resizeImage)
			applyResizeEffect(queryImage);  // Resize Image Effect

		vector<KeyPoint> queryImageKeypoints;
		detectKeypointsImage(queryImage, queryImageKeypoints, featureDetector);

		Mat queryImageDescriptors;
		computeDescriptorsImage(queryImage, queryImageKeypoints, queryImageDescriptors, descriptorExtractor);

	// POINT 2: K-MEANS
		int clusterCount = k;
		while (clusterCount <= numRowsTotal) {

		// POINT 2.1: Apply KMeans on vocabulary (imagesVectorDescriptors)
			vector<vector<int> > labelsVocabularyStructure(clusterCount, vector<int>(numImagesTotal));
			Mat centers;
			cout << "k: " << clusterCount << endl;
			kmeansVocabularyImages(imagesVectorDescriptors, clusterCount, criteriaKMeans, attemptsKMeans, numImagesTotal, labelsVocabularyStructure, centers, numRowsTotal);

		//  POINT 2.2: Find the KCenters on queryImageDescriptors
			Mat kcentersQueryImage(queryImageDescriptors.rows, 1, centers.type());
			findKCentersOnImage(kcentersQueryImage, queryImageDescriptors, centers);

		// POINT 2.3: Voting images (Construct a Mat "matVote" with the number of labels/patterns that contains the query image in our vocabulary)
			Mat matVote = votingImages(labelsVocabularyStructure,kcentersQueryImage,numImagesTotal);

	// POINT 3: RANSAC

			// Create a new imageResult.
			Mat imageResult;
			readImage(queryImageFileName,imageResult,1); // read the image in color to put the result on GREEN and RED

			// POINT 3.1: For every voted image on vocabulary, we select the images with >= "minimumPointsOnVotes" constant
			for (int imag = 0; imag < matVote.rows; ++imag) {
				if (matVote.at<int>(imag,0) >= minimumVotes) {
//					cout << "Image selected: " << imag << " with " << matVote.at<int>(imag,0) << " votes." << endl;

					// POINT 3.2: Select the image and find the KCenters
					Mat imageSelected = vocabularyImages[imag];
					vector<KeyPoint> imageSelectedKeypoints = vocabularyImagesKeypoints[imag];
					Mat imageSelectedDescriptors = imagesVectorDescriptors[imag];
					Mat kcentersImageSelected(imageSelectedDescriptors.rows, 1, centers.type());
					findKCentersOnImage(kcentersImageSelected, imageSelectedDescriptors, centers);

					// POINT 3.3: Apply RANSAC. Look for good/bad homographies. Save result images.
					ransac(kcentersImageSelected, kcentersQueryImage, imageSelected, imageSelectedKeypoints, queryImage, queryImageKeypoints, clusterCount, dirToSaveResImages, imag, thresholdDistanceAdmitted, imageResult, homographyAttempts);
				}
			}
			clusterCount = clusterCount + kIncrement;
		}
	} catch (exception& e) {
		cout << e.what() << endl;
	}
}

int main(int argc, char *argv[]) {

	string algorithmType = "SIFT"; 		// Detector and descriptor type (Ex: "SURF", "SIFT")

	// (Only for SURF):
	bool uprightSURF = false; 			// This is USURF. false=detector computes orientation of each feature. true= the orientation is not computed.
	int hessianThresholdSURF = 500; 	// Threshold for the keypoint detector. A good default value could be from 300 to 500, depending from the image contrast.
	int nOctaves = 4;					//Number of pyramid octaves the keypoint detector will use.
	int nOctaveLayers = 2;				// Number of octave layers within each octave.
	int extended = false;				// Extended descriptor flag (true - use extended 128-element descriptors; false - use 64-element descriptors).

	// Image Effects (Gaussian Blur, resize)
	int kernelSize = -1;				// This means the Gaussian kernel size applied to newImage. (-1: Not apply)
	bool resizeImage = false;			// This means if we make a resize transformation of the image

	// K-Means
	int initialK = 1; 					// Initial K Center constant in k-means. This must be <= Total number of rows in the sum of all vocabulary images.
	int kIncrement = 10;				// This is the increment of the k centers in kmeans loop
	int criteriaKMeans = 100;			// This is the maximum number of iterations in kmeans to recalcule the k-centers (Ex: 100 it's ok)
	int attemptsKMeans = 3;				// This is the number of times the algorithm is executed using different initial labellings (Ex: 3 it's ok)

	// RANSAC
	int minimumVotes = 10;    			// Minimum number of votes that must to have every image to be selected. (Minimum 2.Homography needs 2 points minimum) (Ex: 8-10 are good values)
	int thresholdDistanceAdmitted = 30;	// Threshold distance admitted comparing distance between images on homography results.  (Ex: 30 it's ok)
	int homographyAttempts = 3;			// Number of RANSAC attempts to find homographies

	searchPatterns(algorithmType, hessianThresholdSURF, nOctaves, nOctaveLayers, extended, uprightSURF, initialK, kIncrement, criteriaKMeans, attemptsKMeans, minimumVotes,thresholdDistanceAdmitted, kernelSize, resizeImage, homographyAttempts);
}
