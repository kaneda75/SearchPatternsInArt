#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;
using namespace cv;

void showKeypointsImagesVector(const vector<Mat>& vocabularyImages, const vector<vector<KeyPoint> >& vocabularyImagesKeypoints);
void showKeypointsImage(const Mat& image, const vector<KeyPoint> & imageKeypoints);
void showMatrixValues(Mat& matrix, string s);
void showMatrixValues2(Mat& matrix, string s);
void showMatrixValues3(vector<KeyPoint> keypoints, Mat& matrix,  string s);
int getMostVotedImage(Mat matVote);
Mat createImageResult(const Mat& imageIni, const vector<KeyPoint>& imageIniKeypoints, const Mat& newImage, const vector<KeyPoint>& newImageKeypoints);
