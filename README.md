Search Patterns In Art
======================

This app search patterns in painting images using different computer vision techniques (SIFT/SURF, Kmeans, RANSAC, Homography).

The program reads a set of images predefined by the user. These images contains some patterns (objects) that we want to compare tith other new image . These represent a learning image set or a "vocabulary". Every image of this vocabulary represents a "word". This words will be used by the program to recognize other possible similar patterns in a new image. 


Installation
------------

You need:  
   * C++ Compiler 
   * OpenCV 2.4.3 library (http://opencv.willowgarage.com/wiki/).

License
-------
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

Usage
-----

Once you have downloaded the source code you have to define some CONSTANTS to run the application. These are defined in searchPatterns.cpp:

  
  * **algorithmType** (For example = "SIFT"):   The detector keypoints and type. This can be FAST, STAR, SIFT, SURF, ORB, MSER, GFTT, HARRIS, Dense, SimpleBlob ... 

Only for SURF algorithmType:
  * **uprightSURF** :   This is USURF. false=detector computes orientation of each feature. true= the orientation is not computed. 
  * **hessianThresholdSURF** :  Threshold for the keypoint detector. A good default value could be from 300 to 500, depending from the image contrast.
  * **nOctaves** : Number of pyramid octaves the keypoint detector will use.
  * **nOctaveLayers** :  Number of octave layers within each octave.
  * **extended** :  Extended descriptor flag (true - use extended 128-element descriptors; false - use 64-element descriptors).

Image Effects (Gaussian Blur, resize):
  * **kernelSize** :  This means the Gaussian kernel size applied to newImage. (-1: Not apply)
  * **resizeImage** :  This means if we make a resize transformation of the image

K-Means:
  * **initialK** : Initial K Center constant in k-means. This must be <= Total number of rows in the sum of all vocabulary images.
  * **kIncrement** :  This is the increment of the k centers in kmeans loop
  * **criteriaKMeans** :  This is the maximum number of iterations in kmeans to recalcule the k-centers (Ex: 100 it's ok)
  * **attemptsKMeans** :  This is the number of times the algorithm is executed using different initial labellings (Ex: 3 it's ok)

RANSAC: 
  * **minimumVotes** :  Minimum number of votes that must to have every image to be selected. (Minimum 2.Homography needs 2 points minimum) (Ex: 8-10 are good values)
  * **thresholdDistanceAdmitted** :  Threshold distance admitted comparing distance between images on homography results.  (Ex: 30 it's ok)
  * **homographyAttempts** :  Number of RANSAC attempts to find homographies

Directories, files:
  * **vocabularyImagesNameFile**  (For example = "/../vocabularyImages.txt") This .txt file contains the name of the learning image set Every image represents a "word" inside the "vocabulary" of learning image set
  * **newImageFileName**  (For example = "/../tapies9.jpg"):   This is the new image that we want to compare with the learning image set
  * **dirToSaveResImages**  (For example = "/../results"):  This is the route/directory of to save the result images

