#include <cstdio>
#include <iostream>
#include <math.h>
#include <string.h>
#include "utils.hpp"

using namespace std;
using namespace cv;

void readVocabularyImages(const string& filename, string& dirName, vector<string>& vocabularyFiles) {
	vocabularyFiles.clear();
        ifstream file(filename.c_str());
        if (!file.is_open())
            return;

        size_t pos = filename.rfind('\\');
        char dlmtr = '\\';
        if (pos == String::npos) {
            pos = filename.rfind('/');
            dlmtr = '/';
        }
        dirName = pos == string::npos ? "" : filename.substr(0, pos) + dlmtr;

        while (!file.eof()) {
            string str;
            getline(file, str);
            if (str.empty()) break;
            vocabularyFiles.push_back(str);
        }
        file.close();
}

int leerLineaTxt(FILE *ftxt, char * linea) {
	int error = 0;
	char c = ' ';

	do {
		if (fgets(linea, 120, ftxt) == NULL) {
			error = 1; // ERROR_LECTURA;
		} else {
			c = linea[0];
		};
	} while ((!feof(ftxt)) && (!error)
			&& ((c == '.') || (c == ' ') || (c == '\r') || (c == '\n')
					|| (c == '\0')));
	return (error);
} /* f_leerLinea */


#define Abs(x)    ((x) < 0 ? -(x) : (x))
#define Max(a, b) ((a) > (b) ? (a) : (b))
double RelDif(double a, double b)
{
	double c = Abs(a);
	double d = Abs(b);
	d = Max(c, d);
	return d == 0.0 ? 0.0 : Abs(a - b) / d;
}
