#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string.h>
#include <cstdio>
#include <math.h>
#include <string.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace std;
using namespace cv;

void readVocabularyImages (const string& filename, string& dirName, vector<string>& vocabularyFiles);
int leerLineaTxt(FILE *ftxt, char * linea);
double RelDif(double a, double b);

#endif
