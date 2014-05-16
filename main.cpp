#include <cv.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <highgui.h>
#include <iostream>
#include <random>
#include <map>
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
    net.evaluateValidationSet(validationset, im_handler, 0);
    cout << "Validation error" << net.outputLayer.validationError << "\n";

}

vector<string> split(string line, char delim) {
    stringstream ss(line);
    string s;
    vector<string> tokens;
    while (getline(ss, s, delim)) {
        tokens.push_back(s);
    }
    return tokens;
}

void buildNewDictionary() {

    ifstream infile("/home/silvia/HandwritingRecognition/words.txt");
    if(infile == NULL) {
        cout << "Error when opening file!\n";
        return;
    }
    map< string, string > corpusDict;

    ofstream out("/home/silvia/HandwritingRecognition/corpus/myDictionary");
    if(out == NULL) {
        cout << "Error when opening file!\n";
        return;
    }

    string line;
    while (getline(infile, line))
    {
        vector<string> tokens = split(line, ' ');
//        cout << "1: " << tokens[0] << "\n";
        if(corpusDict.find(tokens[tokens.size()-1]) == corpusDict.end())
        {
            corpusDict.insert(pair<string, string>(tokens[tokens.size()-1], "ok"));
            out << tokens[tokens.size()-1] << "\n";
        }
    }
    out.close();
}

void testDecoding() {

    //use network-recovery constructor
    ImagesHandler im_handler;
    vector<string> trainset = im_handler.getDataSet("trainset.txt");
    vector<string> validationset = im_handler.getDataSet("validationset1.txt");
    NeuralNetwork net("NETWORK_WEIGHTS100LSTM");

    cout << "*TEST*********************************************************\n";
    net.testInputImage(trainset[2], im_handler); //do forward pass through the network for an input example
    cout << "target-label "<< im_handler.getTargetLabel(trainset[2]) << "\n";
    DecodingLayer decodingLayer(net.outputLayer.y,
                                "/home/silvia/HandwritingRecognition/corpus/dictionary",
                                net.outputLayer.alphabet);
    decodingLayer.init();
    vector<string> decodedLabels = decodingLayer.getDecodedLabels();
//    cout << "Decoded labels for word:" << trainset[0] << "\n";
    for(int i = 0; i < decodedLabels.size(); ++i)
        cout << decodedLabels[i] << "\n";

}

void testDecodingOneSet() {

    ImagesHandler im_handler;
    vector<string> trainset = im_handler.getDataSet("trainset.txt");
    vector<string> validationset = im_handler.getDataSet("validationset1.txt");
    ofstream out;
    string filename("DecodedLabels");
    out.open(filename);

    if (!out.is_open())
    {
        cout << "Failed to open file "  << filename << "\n";
        return;
    }

    //use network-recovery constructor
    NeuralNetwork net("NETWORK_WEIGHTS");
    int setSize = 10;
    int correct = 0;

    cout << "*TEST*********************************************************\n";
    for(int i = 0; i < setSize; ++i)
    {
        string target(im_handler.getTargetLabel(validationset[i]));
        net.testInputImage(validationset[i], im_handler); //do forward pass through the network for an input example
        out << target << "\t";
        DecodingLayer decodingLayer(net.outputLayer.y,
                                    "/home/silvia/HandwritingRecognition/corpus/dictionary",
                                    net.outputLayer.alphabet);
        decodingLayer.init();
        vector<string> decodedLabels = decodingLayer.getDecodedLabels();//history
        out << decodedLabels[0] << "\n";
        for(int i = 0; i < decodedLabels.size(); ++i)
            out << decodedLabels[i] << " ";
        out << "\n\n";
        if(decodedLabels[0] == target)
            correct ++;
    }
    cout << "Total correct: " << correct << " of " << setSize << "\n";
    out.close();
}

void debugSmallNetwork() {
    string imagePath("../words/a01/a01-000u/a01-000u-03-01.png");
    NeuralNetwork net(50, 79); //1st arg: #hidden units; 2nd arg: #output_units(ctc)
    net.trainNetworkDebug(imagePath);
}


int main(int argc, char* argv[])
{
    Mat image; //a01-000u-03-02.png
    string imagePath("../words/a01/a01-000u/a01-000u-03-02.png");
    image = imread( imagePath,  CV_LOAD_IMAGE_GRAYSCALE );
    int offset = 0;

    if( !image.data )
    {
        printf( "No image data \n" );
        return -1;
    }

    if(argc > 1) {
        sscanf(argv[1], "%d", &offset);
    }
    cout << "Offset = " << offset << "\n";

    //using random
//    std::default_random_engine generator( time(NULL) );
//    std::normal_distribution<double> distribution (0.0, 0.1);
//    std::cout << "random number" << distribution(generator) << "\n";

//////////////////////////////////////////
//    FeatureExtractor extractor(image);
//    vector< VectorXd > sequenceOfFeatures = extractor.getFeatures();
//    for(uint k = 0; k < sequenceOfFeatures.size(); ++k)
//    {
//        for(unsigned int i = 0; i < NB_FEATURES; ++i)
//            std::cout << sequenceOfFeatures[k][i] << " ";
//        std::cout << "\n";
//    }

//    std::cout << "sequenceSize [T] = " << sequenceOfFeatures.size() << "\n";
//    BLSTM blstm(5, sequenceOfFeatures);
//    blstm.forwardPass();
//    CTCLayer ctc(53, 5);

/***********************************************************************/
//    NeuralNetwork net(50, 79); //1st arg: #hidden units; 2nd arg: #output_units(ctc)
//    net.trainNetwork(0);

    //    net.trainNetworkDebug(imagePath);

    //RETRAIN
//    offset = 4;
//    NeuralNetwork net("NETWORK_WEIGHTS");
//    net.trainNetwork(offset);

    testDecoding(); //one example
//    testDecodingOneSet(); //many examples
//    buildNewDictionary();

    //DEBUG - small net
//    debugSmallNetwork();

    //test DecodingLayer
//    MatrixXd y(10, 79);
//    DecodingLayer dec(y, "/home/silvia/HandwritingRecognition/corpus/dictionary");
//    dec.init();


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
