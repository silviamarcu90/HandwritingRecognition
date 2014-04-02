#include <cv.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <highgui.h>
#include <iostream>
#include <random>
#include <time.h>
#include "../Eigen/Dense"
#include "featureextractor.h"
#include "blstm.h"
#include "ctclayer.h"
#include "neuralnetwork.h"
#include "imageshandler.h"

using namespace cv;
using namespace Eigen;

int main()
{
    Mat image;
    string imagePath("../words/a01/a01-000u/a01-000u-00-02.png");
    image = imread( imagePath,  CV_LOAD_IMAGE_GRAYSCALE );

    if( !image.data )
    {
        printf( "No image data \n" );
        return -1;
    }

    //using random
//    std::default_random_engine generator( time(NULL) );
//    std::normal_distribution<double> distribution (0.0, 0.1);
//    std::cout << "random number" << distribution(generator) << "\n";

//////////////////////////////////////////
    FeatureExtractor extractor(image);
    vector< VectorXd > sequenceOfFeatures = extractor.getFeatures();
//    BLSTM blstm(5, sequenceOfFeatures);
//    blstm.forwardPass();
//    CTCLayer ctc(53, 5);

/***********************************************************************/
    NeuralNetwork net(10, 53);
    net.trainNetwork(imagePath);
/***********************************************************************/
    //list all images-collection
//    ImagesHandler im_handler("../words");
//    vector<string> allImagesPaths = im_handler.getAllFilesList();
//    for(int i = 0; i < allImagesPaths.size(); ++i)
//        cout << allImagesPaths[i] << "\n";
//*********************************************************************/
//    for(unsigned int i = 0; i < 9; ++i)
//        std::cout << sequenceOfFeatures[11][i] << " ";
    std::cout << "\n";
    std::cout << "sequenceSize [T] = " << sequenceOfFeatures.size() << "\n";

//	std::cout << image.depth() << ", " << image.channels() << std::endl;
    Mat im_bw = image >128;

//	int thresh = 128;
//    threshold(image, im_bw, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

//    unsigned char *input = (unsigned char*)(im_bw.data);
//    int imgStep = im_bw.step;
//    std::cout << "Step = " << imgStep << "\n";

//    for(int i = 0; i < image.rows; ++i)
//    {
//        for(int j = 0; j < image.cols; ++j)
//        {
//            std::cout << (int)input[i*imgStep + j] << " ";
//        }
//        std::cout << "\n";
//    }

//    namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
//    imshow( "Display Image", im_bw );

//    waitKey(0);

    return 0;
}
