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
#include "decodinglayer.h"

using namespace cv;
using namespace Eigen;

void testRecoveredNet() {

    //test network - recovery constructor
    ImagesHandler im_handler;
    vector<string> validationset = im_handler.getDataSet("validationset1.txt");
    NeuralNetwork net("NETWORK_WEIGHTS");
    cout << "*********************************************************\n";
    net.evaluateValidationSet(validationset, im_handler);
    cout << "Validation error" << net.outputLayer.validationError << "\n";

}

void testDecoding() {

    //use network-recovery constructor
    ImagesHandler im_handler;
    vector<string> trainset = im_handler.getDataSet("trainset.txt");
    vector<string> validationset = im_handler.getDataSet("validationset1.txt");
    NeuralNetwork net("NETWORK_WEIGHTS");

    cout << "*TEST*********************************************************\n";
    net.testInputImage(trainset[0], im_handler); //do forward pass through the network for an input example

    DecodingLayer decodingLayer(net.outputLayer.y,
                                "/home/silvia/HandwritingRecognition/corpus/dictionary",
                                net.outputLayer.alphabet);
    decodingLayer.init();
    vector<string> decodedLabels = decodingLayer.getDecodedLabels();
    cout << "Decoded labels for word:" << trainset[0] << "\n";
    for(int i = 0; i < decodedLabels.size(); ++i)
        cout << decodedLabels[i] << "\n";
}

int main()
{
    Mat image; //a01-000u-03-02.png
    string imagePath("../words/a01/a01-000u/a01-000u-03-02.png");
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
    for(uint k = 0; k < sequenceOfFeatures.size(); ++k)
    {
        for(unsigned int i = 0; i < NB_FEATURES; ++i)
            std::cout << sequenceOfFeatures[k][i] << " ";
        std::cout << "\n";
    }
//    std::cout << "sequenceSize [T] = " << sequenceOfFeatures.size() << "\n";
//    BLSTM blstm(5, sequenceOfFeatures);
//    blstm.forwardPass();
//    CTCLayer ctc(53, 5);

/***********************************************************************/
//    NeuralNetwork net(10, 79); //1st arg: #hidden units; 2nd arg: #output_units(ctc)
//    net.trainNetwork();

//    NeuralNetwork net("NETWORK_WEIGHTS");
//    net.trainNetwork();

    //test DecodingLayer
//    MatrixXd y(10, 79);
//    DecodingLayer dec(y, "/home/silvia/HandwritingRecognition/corpus/dictionary");
//    dec.init();
//    testDecoding();

//    ImagesHandler im_handler("../words");
//    im_handler.readTargets();
//    im_handler.getTrainingSet();

/***********************************************************************/
    //list all images-collection
//    ImagesHandler im_handler("../words");
//    vector<string> allImagesPaths = im_handler.getAllFilesList();
//    for(int i = 0; i < allImagesPaths.size(); ++i)
//        cout << allImagesPaths[i] << "\n";
//*********************************************************************/

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
