#include "neuralnetwork.h"

void NeuralNetwork::trainNetwork() {

    //TODO: getAllInputs--- FileHandler --- to read the training set -- and transform images into sequences of features
    //[29 March] currently, I read only one image and get the features
    int nbExamples;
    ImagesHandler im_handler;
    vector<string> trainset = im_handler.getTrainingSet();
    nbExamples = 10;//trainset.size();

    cout << "trainset_size: " << nbExamples << "\n";

    //in a loop -- train the weights until a stop condition is fullfilled
    for(int eg = 0; eg < nbExamples; ++eg)
    {
        string imagePath(trainset[eg]);
        FeatureExtractor extractor(imagePath);
        vector< VectorXd > sequenceOfFeatures = extractor.getFeatures();
        string label = im_handler.getTargetLabel(imagePath);
        cout << imagePath << "; label =" << label << "=" << endl;

        trainOneExample( sequenceOfFeatures, label );
    }
}

void NeuralNetwork::trainOneExample(vector<VectorXd> x, string label) {

    inputs = x;

    /// forward pass
    forwardHiddenLayer.forwardPass(inputs); //for each input sequence (image with a word)
    backwardHiddenLayer.forwardPass(inputs);
    outputLayer.forwardPass(inputs.size(), label, forwardHiddenLayer.b_c, backwardHiddenLayer.b_c);

    /////////DEBUG//////////////
//    cout << "FORWARD - layer\n";
//    forwardHiddenLayer.print();
//    cout << "BACKWARD - layer\n";
//    backwardHiddenLayer.print();
//    ////////////////////////////

    /// backward pass
    outputLayer.backwardPass();
    vector<MatrixXd> eps_c1 = outputLayer.getEpsilonCTC();
    forwardHiddenLayer.backwardPass(eps_c1[0]);
    backwardHiddenLayer.backwardPass(eps_c1[1]);

    /// update weights
    outputLayer.updateWeights(ETA);
    forwardHiddenLayer.updateWeights(ETA);
    backwardHiddenLayer.updateWeights(ETA);

    cout << "AFTER FORWARD-pass -- one example\n";
    cout << "#############################################################\n";


    //DEBUG - gradient!
//    double epsilon = 10e-3;
//    outputLayer.w[0](1, 1) += epsilon;
//    outputLayer.forwardPass(inputs.size(), label, forwardHiddenLayer.b_c, backwardHiddenLayer.b_c);
//    outputLayer.backwardPass();

//    outputLayer.w[0](1, 1) -= 2*epsilon;
//    outputLayer.forwardPass(inputs.size(), label, forwardHiddenLayer.b_c, backwardHiddenLayer.b_c);
//    outputLayer.backwardPass();

}

//test network -- implement alg Token - passing

