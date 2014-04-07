/*
 * FeatureExtractor.h
 *
 *  Created on: Mar 14, 2014
 *      Author: silvia
 */

#ifndef FEATUREEXTRACTOR_H_
#define FEATUREEXTRACTOR_H_
#include <cv.h>
#include <highgui.h>
#include "../Eigen/Dense"

using namespace cv;
using namespace Eigen;
using namespace std;

class FeatureExtractor {
    Mat image; /// original image
    Mat im_bw; /// the original image thresholded to black and white
public:
    FeatureExtractor(string imagePath);
    FeatureExtractor(Mat img);
    virtual ~FeatureExtractor();
    vector< VectorXd > getFeatures();
private:
    vector<double> computeFeaturesPerWindow(int winNb);
    VectorXd convertToVectorXd(vector<double> vec);
    double getGradient(int i, int j);
    vector< VectorXd > normalizeFeatures(vector< VectorXd > featuresSeq);
};

#endif /* FEATUREEXTRACTOR_H_ */
