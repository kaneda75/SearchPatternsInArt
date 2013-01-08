#include "utils.hpp"
#include "utilsCV.hpp"

using namespace cv;
using namespace std;

const int color = 0;
int numImagesTotal = 0;

// Directories, files
const string vocabularyImagesNameFile = "/Users/xescriche/git/SearchPatternsInArt/tests/test2/vocabularyImages.txt";
const string newImageFileName = "/Users/xescriche/git/SearchPatternsInArt/tests/test2/tapies4.jpg";
const string dirToSaveResImages = "/Users/xescriche/git/SearchPatternsInArt/tests/test2/results4";

void searchPatterns(string algorithmType, int k, int kIncrement, int criteriaKMeans, int attemptsKMeans, int minimumPointsOnVotes, int thresholdDistanceAdmitted) {
	try {

		// POINT 1: DEFINE Feature detector (detectorType) AND Descriptor extractor (descriptorType)
		Ptr<FeatureDetector> featureDetector = FeatureDetector::create(algorithmType);
		if (featureDetector.empty()) cout << "The detector cannot be created." << endl << ">" << endl;

		Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create(algorithmType);
		if (descriptorExtractor.empty()) cout << "The descriptor cannot be created." << endl << ">" << endl;

		Ptr<Feature2D> surf = Algorithm::create<Feature2D>("Feature2D.SURF");
		if( surf.empty() )
		    CV_Error(CV_StsNotImplemented, "OpenCV was built without SURF support");
		surf->set("hessianThreshold", 500);
		surf->set("nOctaves", 4);
		surf->set("nOctaveLayers", 2);
		surf->set("upright", false);
		surf->set("extended",true);
		featureDetector = surf;


		// POINT 2.1: READ IMAGES, DETECT KEYPOINTS AND EXTRACT DESCRIPTORS ON "VOCABULARY IMAGES"
		vector<Mat> vocabularyImages;
		vector<string> vocabularyImagesNames;
		if (!readImagesFromFile(vocabularyImagesNameFile, vocabularyImages,vocabularyImagesNames, color, numImagesTotal)) cout << endl;

		vector<vector<KeyPoint> > vocabularyImagesKeypoints;
		detectKeypointsImagesVector(vocabularyImages, vocabularyImagesKeypoints, featureDetector);

		vector<Mat> imagesVectorDescriptors;
		computeDescriptorsImagesVector(vocabularyImages, vocabularyImagesKeypoints,imagesVectorDescriptors, descriptorExtractor);

		// POINT 2.2: KMEANS ON imagesVectorDescriptors
		int numRowsTotal = calculeNumRowsTotal(imagesVectorDescriptors);
		cout << "numRowsTotal: " << numRowsTotal << endl;

		int clusterCount = k;
		while (clusterCount < numRowsTotal) {
			vector<vector<int> > vocabulary(clusterCount, vector<int>(numImagesTotal));
			Mat labels;
			Mat centers;
			cout << endl;
			cout << "k: " << clusterCount << endl;
			kmeansVocabularyImages(imagesVectorDescriptors, clusterCount, criteriaKMeans, attemptsKMeans, numImagesTotal, vocabulary, labels, centers, numRowsTotal);

			// POINT 3.1: READ IMAGE, DETECT KEYPOINTS AND EXTRACT DESCRIPTORS ON "NEW IMAGE"
			Mat newImage;
			if (!readImage(newImageFileName,newImage,color))
				cout << endl;

			vector<KeyPoint> newImageKeypoints;
			detectKeypointsImage(newImage, newImageKeypoints, featureDetector);

			Mat newImageDescriptors;
			computeDescriptorsImage(newImage, newImageKeypoints, newImageDescriptors, descriptorExtractor);

			//  POINT 3.2: Find Words/KCenters on newImageDescriptors
			Mat wordsNewImage(newImageDescriptors.rows, 1, centers.type());
			findKCentersOnNewImage(wordsNewImage, newImageDescriptors, centers);

			// POINT 3.3: Voting images
			Mat matVote = votingImages(vocabulary,wordsNewImage,numImagesTotal);
			for (int imag = 0; imag < matVote.rows; ++imag) {
				if (matVote.at<int>(imag,0) >= minimumPointsOnVotes) {
					cout << endl;
					cout << "Image selected: " << imag << " with " << matVote.at<int>(imag,0) << " votes." << endl;
					// POINT 4.1: RANSAC
					Mat imageSelected = vocabularyImages[imag];
					vector<KeyPoint> imageSelectedKeypoints = vocabularyImagesKeypoints[imag];
					Mat imageSelectedDescriptors = imagesVectorDescriptors[imag];
					Mat wordsImageIni(imageSelectedDescriptors.rows, 1, centers.type());
					findKCentersOnNewImage(wordsImageIni, imageSelectedDescriptors, centers);
					ransac(wordsImageIni, wordsNewImage, imageSelected, imageSelectedKeypoints, newImage, newImageKeypoints, clusterCount, dirToSaveResImages, imag, thresholdDistanceAdmitted);
				}
			}
			clusterCount = clusterCount + kIncrement;
		}
	} catch (exception& e) {
		cout << e.what() << endl;
	}
}

int main(int argc, char *argv[]) {

	string algorithmType = "SURF";		// Detector and descriptor type (Ex: "SURF", "SIFT")
	int initialK = 1; 					// Initial K Center constant in k-means. This must be <= Total number of rows in the sum of all vocabulary images.
	int kIncrement = 10;				// This is the increment of the k centers in kmeans loop
	int minimumPointsOnVotes = 10;    	// Minimum number of votes that must to have every image to be selected. (Minimum 2.Homography needs 2 points minimum) (Ex: 8-10 are good values)
	int criteriaKMeans = 100;			// This is the maximum number of iterations in kmeans to recalcule the k-centers (Ex: 100 it's ok)
	int attemptsKMeans = 3;				// This is the number of times the algorithm is executed using different initial labellings (Ex: 3 it's ok)
	int thresholdDistanceAdmitted = 3;	// Threshold distance admitted comparing distance between images on homography results.  (Ex: 3 it's ok)

	searchPatterns(algorithmType, initialK, kIncrement, criteriaKMeans, attemptsKMeans, minimumPointsOnVotes,thresholdDistanceAdmitted);
}
