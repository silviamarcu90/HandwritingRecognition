#ifndef IMAGESHANDLER_H
#define IMAGESHANDLER_H

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <map>

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

class ImagesHandler
{
private:
    void listFilesOfDir(string dir, vector<string> &allFiles);
    vector<string> split(string line, char delim);

public:
    ImagesHandler();
    vector<string> getAllFilesList();
    void createMapImgTargets();
    vector<string> getDataSet(string setTypeFile);
    string getTargetLabel(string imagePath);

    vector<string> allFilenames;
    map< string, string > mapImgTarget; /// map containing pairs with key=image_path and value=corresponding, i.e word
//    vector< string> trainset;
    string dirPath;
};

#endif // IMAGESHANDLER_H
