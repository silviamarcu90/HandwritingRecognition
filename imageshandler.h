#ifndef IMAGESHANDLER_H
#define IMAGESHANDLER_H

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

class ImagesHandler
{
private:
    void listFilesOfDir(string dir);

public:
    ImagesHandler(string dirPath);
    vector<string> getAllFilesList();

    vector<string> allFilenames;
    string dirPath;
};

#endif // IMAGESHANDLER_H
