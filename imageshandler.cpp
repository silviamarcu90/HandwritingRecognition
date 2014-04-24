#include "imageshandler.h"

ImagesHandler::ImagesHandler()
{
    createMapImgTargets();
}

vector<string> ImagesHandler::getAllFilesList() {
    dirPath = "../words";
    listFilesOfDir(dirPath, allFilenames);
    return allFilenames;
}

void ImagesHandler::listFilesOfDir(string dir, vector<string>& allFiles) {
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
            listFilesOfDir(filepath, allFiles);
        else {// is an expected image
//            cout << filepath << "\n";
            allFiles.push_back(filepath);
        }
    }

    closedir( dp );

}

/**
 * Function used to read corresponding transcriptions of the images
 * and to create a dictionary <key=imageName, value=transcription>
 */
void ImagesHandler::createMapImgTargets() {

    ifstream infile("/home/silvia/HandwritingRecognition/words.txt");
    if(infile == NULL) {
        cout << "Error when opening file!\n";
        return;
    }
    int k = 0;
    string line;
    while (getline(infile, line))
    {
        vector<string> tokens = split(line, ' ');
//        cout << "1: " << tokens[0] << "; " << tokens[tokens.size()-1] << "\n";
        mapImgTarget.insert(pair<string, string>(tokens[0], tokens[tokens.size()-1]));
        k++;
//        if(k == 10000)
//            break;
    }
}

vector<string> ImagesHandler::split(string line, char delim) {
    stringstream ss(line);
    string s;
    vector<string> tokens;
    while (getline(ss, s, delim)) {
        tokens.push_back(s);
    }
    return tokens;
}

/**
 * Read trainset file with the set distributed for training
 * in order to be used further for training the network
 */
vector<string> ImagesHandler::getDataSet(string setTypeFile) {
    vector< string> dataset;
    string prefixPath("../words/");
    ifstream infile("/home/silvia/HandwritingRecognition/setsDistribution/" + setTypeFile);
    if(infile == NULL) {
        cout << "Error when opening file!\n";
    }

    int k = 0;
    string line;
    string prevSuffix = "";
    cout << "Start parsing file: " + setTypeFile + "\n";
    while (getline(infile, line)) {

        vector<string> tmpSet;
        vector<string> tokens = split(line, '-');
        string suffix(tokens[0]+"/" + tokens[0] + "-" + tokens[1]);
        if(suffix.compare(prevSuffix) == 0) continue;
//        cout << "1: " << tokens[0] << "; " << tokens[1] << "\n";
        listFilesOfDir(prefixPath + suffix, tmpSet);

        dataset.insert(dataset.end(), tmpSet.begin(), tmpSet.end());
        k++;
        if(k == 2) //extract only a part of the training set, i.e one subfolder
            break;
        prevSuffix = suffix;
    }

//    for(int i = 0; i < trainset.size(); ++i)
//        cout << trainset[i] << "\n";
//    cout << trainset.size() << "\n";
    return dataset;
}

string ImagesHandler::getTargetLabel(string imagePath) {

    unsigned found = imagePath.find_last_of("/");
    string imageName = imagePath.substr(found+1);
    found = imageName.find_last_of(".");
    string key = imageName.substr(0, found);

    return mapImgTarget[key];
}

