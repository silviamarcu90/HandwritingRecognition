#include "neuralnetwork.h"

void NeuralNetwork::trainNetwork() {

    int nbExamples;
    ImagesHandler im_handler;
    vector<string> trainset = im_handler.getDataSet("trainset.txt");
    vector<string> validationset = im_handler.getDataSet("validationset.txt");
    nbExamples = 50;//trainset.size();
    double prevValidationError = 10e5;

    cout << "trainset_size: " << trainset.size() << "\n";
    cout << "validationset_size: " << validationset.size() << "\n";

    out.open("trainErrors20Egs.txt");

    if (!out.is_open())
    {
        cout << "Failed to open file...\n";
        return;
    }
    //in a loop -- train the weights until a stop condition is fullfilled
    for(int epoch = 0; epoch < 10/*MAX_ITER*/; ++ epoch)
    {

        outputLayer.trainError = 0.0; // the training error
        outputLayer.validationError = 0.0;
        for(int eg = 0; eg < nbExamples; ++ eg)
        {
            string imagePath(trainset[eg]);
            FeatureExtractor extractor(imagePath);
            vector< VectorXd > sequenceOfFeatures = extractor.getFeatures();
            string label = im_handler.getTargetLabel(imagePath);
            cout << imagePath << "; label =" << label << "=" << endl;

            trainOneExample( sequenceOfFeatures, label );
        }
        cout << "*********************************************************\n";
        evaluateValidationSet(validationset, im_handler);
        out << outputLayer.trainError << " ";
        out << outputLayer.validationError << "\n";
        if(epoch % 5 == 0)
            prevValidationError = outputLayer.validationError;

        if(epoch % 5 == 0 && prevValidationError > outputLayer.validationError)
            break;
    }

    out.close();
}

void NeuralNetwork::trainOneExample(vector<VectorXd> x, string label) {

    inputs = x;

    /// forward pass
    forwardHiddenLayer.forwardPass(inputs); //for each input sequence (image with a word)
    backwardHiddenLayer.forwardPass(inputs);
    outputLayer.forwardPass(inputs.size(), label, forwardHiddenLayer.b_c, backwardHiddenLayer.b_c);

//    /////////DEBUG//////////////
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

    cout << "AFTER passing -- one example\n";
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

/**
 * validate network
 */
void NeuralNetwork::evaluateValidationSet(vector<string> validationset, ImagesHandler im_handler) {

    cout << "VALIDATE!\n";
    int setSize = 50;//validationset.size();

    for(int i = 0; i < setSize; ++ i)
    {
        string imagePath(validationset[i]);
        FeatureExtractor extractor(imagePath);
        vector< VectorXd > sequenceOfFeatures = extractor.getFeatures();
        string label = im_handler.getTargetLabel(imagePath);
        cout << imagePath << "; label =" << label << "=" << endl;
        inputs = sequenceOfFeatures;

        /// forward pass
        forwardHiddenLayer.forwardPass(inputs);
        backwardHiddenLayer.forwardPass(inputs);
        outputLayer.forwardPass(inputs.size(), label, forwardHiddenLayer.b_c, backwardHiddenLayer.b_c);

        /// compute error
        outputLayer.validationError += outputLayer.computeError();
    }

}
//test network -- implement alg Token - passing

