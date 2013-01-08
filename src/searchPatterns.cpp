#include "utils.hpp"
#include "utilsCV.hpp"

using namespace cv;
using namespace std;

// Detectors, descriptors, loadImage
const int color = 0;
int numImagesTotal = 0;

// Directories, files
const string vocabularyImagesNameFile = "/Users/xescriche/git/SearchPatternsInArt/tests/test2/vocabularyImages.txt";
const string newImageFileName = "/Users/xescriche/git/SearchPatternsInArt/tests/test2/tapies1.jpg";
const string dirToSaveResImages = "/Users/xescriche/git/SearchPatternsInArt/tests/test2/results4";

void searchPatterns(string algorithmType, int k, int criteriaKMeans, int attemptsKMeans, int minimumPointsOnVotes) {
	try {

		// POINT 1: DEFINE Feature detector (detectorType) AND Descriptor extractor (descriptorType)

		// Feature detector
		Ptr<FeatureDetector> featureDetector;
		featureDetector = FeatureDetector::create(algorithmType);
		if (featureDetector.empty())
			cout << "The detector cannot be created." << endl << ">" << endl;

		// Descriptor extractor
		Ptr<DescriptorExtractor> descriptorExtractor;
		descriptorExtractor = DescriptorExtractor::create(algorithmType);
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
		//		 showKeypointsImagesVector(vocabularyImages, vocabularyImagesKeypoints);

		vector<Mat> imagesVectorDescriptors;
		computeDescriptorsImagesVector(vocabularyImages, vocabularyImagesKeypoints,imagesVectorDescriptors, descriptorExtractor);

		// POINT 2.2: KMEANS ON imagesVectorDescriptors

		int numRowsTotal = calculeNumRowsTotal(imagesVectorDescriptors);
		cout << "numRowsTotal: " << numRowsTotal << endl;

		for (int clusterCount = k; clusterCount <= numRowsTotal; ++clusterCount) {

			vector<vector<int> > vocabulary(clusterCount, vector<int>(numImagesTotal));
			Mat labels;
			Mat centers;

			cout << "k: " << clusterCount << endl;
			kmeansVocabularyImages(imagesVectorDescriptors, clusterCount, criteriaKMeans, attemptsKMeans, numImagesTotal, vocabulary, labels, centers, numRowsTotal);

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
			findKCentersOnNewImage(wordsNewImage, newImageDescriptors, centers);

			// POINT 3.3: Voting images

			Mat matVote = votingImages(vocabulary,wordsNewImage,numImagesTotal);
			//		int mostVotedImage = getMostVotedImage(matVote);
			for (int imag = 0; imag < matVote.rows; ++imag) {
				if (matVote.at<int>(imag,0) >= minimumPointsOnVotes) {

					cout << "Image selected: " << imag << " with " << matVote.at<int>(imag,0) << " votes." << endl;
					// POINT 4.1: RANSAC

					Mat imageSelected = vocabularyImages[imag];
					vector<KeyPoint> imageSelectedKeypoints = vocabularyImagesKeypoints[imag];
					Mat imageSelectedDescriptors = imagesVectorDescriptors[imag];
					Mat wordsImageIni(imageSelectedDescriptors.rows, 1, centers.type());
					findKCentersOnNewImage(wordsImageIni, imageSelectedDescriptors, centers);

					//		showMatrixValues3(imageSelectedKeypoints,wordsImageIni, "wordsImageIni:");
					//		showMatrixValues3(newImageKeypoints, wordsNewImage, "wordsNewImage:");
					//		showKeypointsImage(imageSelected, imageSelectedKeypoints);
					//		showKeypointsImage(newImage, newImageKeypoints);

					ransac(wordsImageIni, wordsNewImage, imageSelected, imageSelectedKeypoints, newImage, newImageKeypoints, clusterCount, dirToSaveResImages, imag);

				}
			}
		}
	} catch (exception& e) {
		cout << e.what() << endl;
	}
}

int main(int argc, char *argv[]) {

	string algorithmType = "SIFT";
	int k = 1; 						  // K const in k-means. This must be <= Total number of rows in the sum of all vocabulary images.
	int minimumPointsOnVotes = 10;    // This must be minimum 2. Homography needs 2 points minimum. 8-10 is a good value
	int criteriaKMeans = 100;
	int attemptsKMeans = 3;

	searchPatterns(algorithmType, k, criteriaKMeans, attemptsKMeans, minimumPointsOnVotes);
}
