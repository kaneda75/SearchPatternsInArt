#include <cstdio>
#include <iostream>
#include <math.h>
#include <string.h>
#include "utils.hpp"

using namespace std;
using namespace cv;

static void readVocabularyImages(const string& filename, string& dirName, vector<string>& vocabularyFiles) {
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
