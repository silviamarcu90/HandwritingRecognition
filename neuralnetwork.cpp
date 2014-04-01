#include "neuralnetwork.h"

//NeuralNetwork::NeuralNetwork(int hiddenUnitsNum, int outputUnitsNum)
//{
//    cout << "Constructor - NeuralNetwork\n";
//    BLSTM fhl(hiddenUnitsNum);
//    forwardHiddenLayer = fhl;
//    backwardHiddenLayer = fhl;
//    CTCLayer ctc(outputUnitsNum, hiddenUnitsNum);
//    outputLayer = ctc;
//}


void NeuralNetwork::trainNetwork() {

    //TODO: getAllInputs--- FileHandler --- to read the training set -- and transform images into sequences of features
    //[29 March] currently, I read only one image and get the features
    int nbExamples = 1;

    //in a loop -- train the weights until a stop condition is fullfilled
    for(int eg = 0; eg < nbExamples; ++eg)
    {
        string imagePath("../words/a01/a01-000u/a01-000u-00-01.png");
        FeatureExtractor extractor(imagePath);
        vector< VectorXd > sequenceOfFeatures = extractor.getFeatures();

        trainExample( sequenceOfFeatures, "MOVE");
    }
}

void NeuralNetwork::trainExample(vector<VectorXd> x, string label) {

    inputs = x;
    /// forward pass
    forwardHiddenLayer.forwardPass(inputs); //for each input sequence (image with something written)
    backwardHiddenLayer.forwardPass(inputs);
    outputLayer.forwardPass(inputs.size(), label, forwardHiddenLayer.b_c, backwardHiddenLayer.b_c); //!!! the second arg is for the backward layer

    /////////////////////////////
    cout << "FORWARD - layer\n";
    forwardHiddenLayer.print();
    cout << "BACKWARD - layer\n";
    backwardHiddenLayer.print();
    ////////////////////////////

//    /// backward pass
//    outputLayer.backwardPass();
//    vector<MatrixXd> eps_c1 = outputLayer.getEpsilonCTC();
//    forwardHiddenLayer.backwardPass(eps_c1[0]);
//    backwardHiddenLayer.backwardPass(eps_c1[1]); // the same argument for the backward-hidden layer

//    /// update weights
//    outputLayer.updateWeights(ETA);
//    forwardHiddenLayer.updateWeights(ETA);
//    backwardHiddenLayer.updateWeights(ETA);

}

//test network -- implement alg Token - passing

