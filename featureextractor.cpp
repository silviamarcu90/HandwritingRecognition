/*
 * FeatureExtractor.cpp
 *
 *  Created on: Mar 14, 2014
 *      Author: silvia
 */

#include "featureextractor.h"

FeatureExtractor::FeatureExtractor(string imagePath) {
    image = imread( imagePath,  CV_LOAD_IMAGE_GRAYSCALE );
    im_bw = image >128;
}

FeatureExtractor::FeatureExtractor(Mat img) {
    image = img;
    im_bw = image >128; // TODO: use CV_THRESH_OTSU
}

double FeatureExtractor::computeRateOfChange(vector<vector <double> > sequenceOfFeatures, int j, int featurePos)
{
    int neighboursCount = 0;
    double prev = 0.0, next = 0.0, avgDiff;
    double current = sequenceOfFeatures[j][featurePos];
    current = (current == 0) ? 1 : current;
    if(j - 1 >= 0)
    {
        prev = abs(sequenceOfFeatures[j-1][featurePos] - current);
        neighboursCount ++;
    }
    if(j + 1 < im_bw.cols)
    {
        next = abs(sequenceOfFeatures[j+1][featurePos] - current);
        neighboursCount ++;
    }
    neighboursCount = (neighboursCount == 0) ? 1 : neighboursCount;
    avgDiff = (prev + next)/neighboursCount;
    return avgDiff/current;
}

/**
 * @return the sequence of vectors of features of the whole image
 */
vector< VectorXd > FeatureExtractor::getFeatures(){
    int cols = im_bw.cols;
    vector< vector <double> > sequenceOfFeatures;
    vector< VectorXd > result;

    double f8, f9;
    for(int j = 0; j < cols; ++j)
    {
        vector<double> winFeatures = computeFeaturesPerWindow(j);
        if(j == 10)
            std::cout << winFeatures.size() << std::endl;
        sequenceOfFeatures.push_back(winFeatures);
        //I need to compute the rate of change
        //of the lower and the uppermost position with respect to the neighbors
        if(j > 0)
        {
            //f4 = uppermost -- compute correspondent changing rate
            f8 = computeRateOfChange(sequenceOfFeatures, j-1, 3);
            //f5 = lowermost -- compute correspondent changing rate
            f9 = computeRateOfChange(sequenceOfFeatures, j-1, 4);
            sequenceOfFeatures[j-1].push_back(f8);
            sequenceOfFeatures[j-1].push_back(f9);
            result.push_back(convertToVectorXd(sequenceOfFeatures[j-1]));
        }
    }
    for(unsigned int i = 0; i < 9; ++i)
        std::cout << sequenceOfFeatures[10][i] << " ";
    std::cout << "\n";
    //compute the additional features - changing rate - for the last column
    f8 = computeRateOfChange(sequenceOfFeatures, cols-1, 3);
    f9 = computeRateOfChange(sequenceOfFeatures, cols-1, 4);
    sequenceOfFeatures[cols-1].push_back(f8);
    sequenceOfFeatures[cols-1].push_back(f9);
    result.push_back(convertToVectorXd(sequenceOfFeatures[cols-1]));

    return result;
}


VectorXd FeatureExtractor::convertToVectorXd(vector<double> vec)
{
    VectorXd res;
    int n = vec.size();
    res = VectorXd::Zero(n);
    for(int i = 0; i < n; ++i)
        res[i] = vec[i];
    return res;
}

vector<double> FeatureExtractor::computeFeaturesPerWindow(int winNb)
{
    unsigned char *input = (unsigned char*)(im_bw.data);
    int imgStep = im_bw.step;
    int j = winNb; //the column for which the features are computed
    int rows = im_bw.rows;
    double f1 = 0.0, f2 = 0.0, f3 = 0.0;
    vector<double> featuresVec;

    //compute the first 3 features:
    // f1: number of black pixels (= mean gray value)
    // f2: center of gravity
    // f3: second order moment of the window
    for(int i = 0; i < rows; ++i)
    {
        int pixelVal = (int)input[i*imgStep + j]/255;
        f1 += pixelVal;
        f2 += i*pixelVal;
        f3 += i*i*pixelVal;
    }
    f1 /= rows;
    f2 /= rows;
    f3 = f3/(rows*rows);

    //DEBUG
    if(winNb == 10)
    {
        std::cout << "f1 " << f1 << "\n";
        std::cout << "f2 " << f2 << "\n";
        std::cout << "f3 " << f3 << "\n";
    }
    featuresVec.push_back(f1);
    featuresVec.push_back(f2);
    featuresVec.push_back(f3);

    int uppermost = 0, lowermost = rows - 1;
    //f4: compute position of the uppermost black pixel
    for(int i = 0; i < rows; ++i)
        if((int)input[i*imgStep + j] == 0)
        {
            uppermost = i;
            break;
        }
    //f5: compute position of the lowermost black pixel
    for(int i = rows-1; i >= 0; i--)
        if((int)input[i*imgStep + j] == 0)
        {
            lowermost = i;
            break;
        }
    featuresVec.push_back(uppermost);
    featuresVec.push_back(lowermost);

    //f6: number of black-white transitions between the uppermost and the lowermost
    //f7: proportion of black pixels in this region
    int f6 = 0, prevPixelVal = 0; //the current pixel is black
    double f7 = 1.0; // the pixel corresponding to the uppermost
    int interval = lowermost - uppermost + 1;
    for(int i = uppermost + 1; i <= lowermost; ++i)
    {
        int pixelVal = (int)input[i*imgStep + j]/255;
        if(pixelVal != prevPixelVal)
            f6++;
        prevPixelVal = pixelVal;
        if(pixelVal == 0)
            f7++;
    }
    if(interval != 0)
        f7 /= interval;
    featuresVec.push_back(f6);
    featuresVec.push_back(f7);
//	if(winNb == 10)
//	{
//		for(unsigned int i = 0; i < featuresVec.size(); ++i)
//			std::cout << featuresVec[i] << " ";
//		std::cout << "\n";
//	}

    return featuresVec;
}

FeatureExtractor::~FeatureExtractor() {
    // TODO Auto-generated destructor stub
}
