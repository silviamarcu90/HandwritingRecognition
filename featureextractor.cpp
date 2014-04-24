/*
 * FeatureExtractor.cpp
 *
 *  Created on: Mar 14, 2014
 *      Author: silvia
 */

#include "featureextractor.h"

FeatureExtractor::FeatureExtractor(string imagePath) {
    image = imread( imagePath,  CV_LOAD_IMAGE_GRAYSCALE );
    threshold(image, im_bw, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
}

FeatureExtractor::FeatureExtractor(Mat img) {
    image = img;
    threshold(image, im_bw, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

}

/**
 * Function used to compute the features of the whole image
 * @return the sequence of vectors of features
 */
vector< VectorXd > FeatureExtractor::getFeatures(){
    int cols = im_bw.cols;
    vector< vector <double> > sequenceOfFeatures;
    vector< VectorXd > result;

    for(int j = 0; j < cols; ++j)
    {
        vector<double> winFeatures = computeFeaturesPerWindow(j);
        sequenceOfFeatures.push_back(winFeatures);
        result.push_back(convertToVectorXd(sequenceOfFeatures[j]));
    }

    return normalizeFeatures(result);
}

/**
 * Normalization of the features using the standardization method
 */
vector< VectorXd > FeatureExtractor::normalizeFeatures(vector< VectorXd > featuresSeq){
    int T = featuresSeq.size();
    int numFeatures = featuresSeq[0].size();
    VectorXd mean = VectorXd::Zero(numFeatures);
    VectorXd std_dev = VectorXd::Zero(numFeatures);

    //1. compute mean
    for(int i = 0; i < T; ++i)
        mean += featuresSeq[i];

    for(int i = 0; i < numFeatures; ++i)
        mean(i) /= T;

    //2. compute standard deviation
    for(int i = 0; i < T; ++i) {
        VectorXd diff = featuresSeq[i] - mean;
        for(int j = 0; j < numFeatures; ++j)
            diff(j) = diff(j) * diff(j);
        std_dev += diff;
    }
    for(int i = 0; i < numFeatures; ++i)
        std_dev(i) = sqrt( std_dev(i)/T );

    //3. normalize features
    for(int i = 0; i < T; ++i) {
        for(int j = 0; j < numFeatures; ++j) {
            featuresSeq[i](j) = (featuresSeq[i](j) - mean(j));
            if(std_dev(j) != 0)
                featuresSeq[i](j) /= std_dev(j);
//            cout << featuresSeq[i](j) << " ";
        }
//        cout << "\n";
    }

    return featuresSeq;
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

    //f6: the rate of change of the uppermost position
    //f7: the rate of change of the lowermost position
    double f6 = getGradient(uppermost, j);
    double f7 = getGradient(lowermost, j);

    featuresVec.push_back(uppermost);
    featuresVec.push_back(lowermost);
    featuresVec.push_back(f6);
    featuresVec.push_back(f7);

    //f8: number of black-white transitions between the uppermost and the lowermost
    //f9: proportion of black pixels in this region
    int f8 = 0, prevPixelVal = 0; //the current pixel is black
    double f9 = 1.0; // the pixel corresponding to the uppermost
    int interval = lowermost - uppermost + 1;
    for(int i = uppermost + 1; i <= lowermost; ++i)
    {
        int pixelVal = (int)input[i*imgStep + j]/255;
        if(pixelVal != prevPixelVal)
            f8++;
        prevPixelVal = pixelVal;
        if(pixelVal == 0)
            f9++;
    }
    if(interval != 0)
        f9 /= interval;
    featuresVec.push_back(f8);
    featuresVec.push_back(f9);

    return featuresVec;
}

double FeatureExtractor::getGradient(int i, int j) {

    unsigned char *input = (unsigned char*)(im_bw.data);
    int cols = im_bw.cols;
    int rows = im_bw.rows;
    int imgStep = im_bw.step;

    double prev_x, next_x, diff_x;
    double prev_y, next_y, diff_y;

    prev_x = (j > 0) ? (int)input[i*imgStep + (j-1)]/255 : 0;
    next_x = (j < cols - 1) ? (int)input[i*imgStep + (j+1)]/255 : 0;

    prev_y = (i > 0) ? (int)input[(i-1)*imgStep + j]/255 : 0;
    next_y = (i < rows - 1) ? (int)input[(i+1)*imgStep + j]/255 : 0;

    diff_x = (next_x - prev_x)/2;
    diff_y = (next_y - prev_y)/2;

    return sqrt(diff_x*diff_x + diff_y*diff_y);
}

FeatureExtractor::~FeatureExtractor() {
    image.release();
    im_bw.release();
}
