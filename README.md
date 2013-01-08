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

  * **detectorType** (For example = "SIFT"):   The detector keypoints type. This can be FAST, STAR, SIFT, SURF, ORB, MSER, GFTT, HARRIS, Dense, SimpleBlob ... 
http://opencv.willowgarage.com/documentation/cpp/features2d_common_interfaces_of_feature_detectors.html

  * **descriptorType** (For example = "SIFT"):   The descriptror extractor type. This can be SIFT, SURF, ORB, BRIEF,... 
http://opencv.willowgarage.com/documentation/cpp/features2d_common_interfaces_of_descriptor_extractors.html

  * **color** (For example = 0):   This is the color mode at read images. 0) CV_LOAD_IMAGE_GRAYSCALE. http://opencv.willowgarage.com/documentation/c/reading_and_writing_images_and_video.html  

  * **clusterCount** (For example = 282): This is the K constant used in k-means algorithm to define the number of k centers. This must be <= Total number of rows in the sum of all vocabulary images
  
  * **attempts** (For example = 3):   This is the number of attempts that uses kmeans to recalcule the k centers
  
  * **vocabularyImagesNameFile**  (For example = "/../vocabularyImages.txt") This .txt file contains the name of the learning image set Every image represents a "word" inside the "vocabulary" of learning image set
  
  * **newImageFileName**  (For example = "/../tapies9.jpg"):   This is the new image that we want to compare with the learning image set
  
  * **dirToSaveResImages**  (For example = "/../results"):  This is the route/directory of to save the result images
