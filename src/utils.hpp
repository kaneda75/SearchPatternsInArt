#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/contrib/contrib.hpp"

using namespace std;
using namespace cv;

void readVocabularyImages (const string& filename, string& dirName, vector<string>& vocabularyFiles);
int leerLineaTxt(FILE *ftxt, char * linea);

#endif
