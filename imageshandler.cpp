#include "imageshandler.h"

ImagesHandler::ImagesHandler(string dirPath)
{
    this->dirPath = dirPath;
}

vector<string> ImagesHandler::getAllFilesList() {
    listFilesOfDir(dirPath);
    return allFilenames;
}

void ImagesHandler::listFilesOfDir(string dir) {
    string filepath;
    DIR *dp;
    struct dirent *dirp;
    struct stat filestat;

    dp = opendir( dir.c_str() );
    if (dp == NULL)
    {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return;
    }

    while ( (dirp = readdir( dp )) )
    {
        filepath = dir + "/" + dirp->d_name;
        string filename(dirp->d_name);

        // If the file is a directory (or is in some way invalid) we'll skip it
        if (stat( filepath.c_str(), &filestat )) continue;
        if(filename.compare("..") == 0 || filename.compare(".") == 0) //not a desired directory
            continue;
        if (S_ISDIR( filestat.st_mode )) //if the file is a subdirectory, list all the files inside
            listFilesOfDir(filepath);
        else {// is an expected image
//            cout << filepath << "\n";
            allFilenames.push_back(filepath);
        }
    }

    closedir( dp );

}
