#include "neuralnetwork.h"

void NeuralNetwork::trainNetwork() {

    int nbExamples;
    ImagesHandler im_handler;
    vector<string> trainset = im_handler.getDataSet("trainset.txt");
    vector<string> validationset = im_handler.getDataSet("validationset1.txt");
    nbExamples = 1;//trainset.size();
    double prevValidationError = 10e5;

    cout << "trainset_size: " << trainset.size() << "\n";
    cout << "validationset_size: " << validationset.size() << "\n";

    out.open("trainErrors1Egs_1ITER.txt");

    if (!out.is_open())
    {
        cout << "Failed to open file...\n";
        return;
    }
    //in a loop -- train the weights until a stop condition is fullfilled
    for(int epoch = 0; epoch < 1/*MAX_ITER*/; ++ epoch)
    {

        outputLayer.trainError = 0.0; // the training error
        outputLayer.validationError = 0.0;
        for(int eg = 0; eg < nbExamples; ++ eg)
        {
            string imagePath(trainset[eg]);
            FeatureExtractor extractor(imagePath);
            vector< VectorXd > sequenceOfFeatures = extractor.getFeatures();
            string label = im_handler.getTargetLabel(imagePath);
//            cout << imagePath << "; label =" << label << "=" << endl;

            trainOneExampleDebug( sequenceOfFeatures, label );
        }
        cout << "*********************************************************\n";
//        evaluateValidationSet(validationset, im_handler);
        out << (epoch+1) << " " << outputLayer.trainError << " ";
        out << outputLayer.validationError << "\n";
//        if(epoch % 5 == 0)
//            prevValidationError = outputLayer.validationError;

//        if(epoch % 5 == 0 && prevValidationError > outputLayer.validationError)
//            break;
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

//    cout << "AFTER passing -- one example\n";
//    cout << "#############################################################\n";

}

void NeuralNetwork::trainOneExampleDebug(vector<VectorXd> x, string label) {

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

//    cout << "AFTER passing -- one example\n";
//    cout << "#############################################################\n";


    //DEBUG - gradient!
    double epsilon = 1e-3;
    int c = 2, i = 1, j = 3;
//    cout << "w[0](1,1) = " << outputLayer.w[0](i,j) << "\n";
//    outputLayer.w[0](i, j) += epsilon;
//    outputLayer.forwardPass(inputs.size(), label, forwardHiddenLayer.b_c, backwardHiddenLayer.b_c);
//    outputLayer.backwardPass();
//    double Ominus = outputLayer.computeObjectiveFunction();
//    cout << "w[0](1,1) = " << outputLayer.w[0](i,j) << "\n";

//    outputLayer.w[0](i, j) -= 2*epsilon; //because I need to subtract the previous addition
//    outputLayer.forwardPass(inputs.size(), label, forwardHiddenLayer.b_c, backwardHiddenLayer.b_c);
//    outputLayer.backwardPass();
//    double Oplus = outputLayer.computeObjectiveFunction();
//    cout << "w[0](1,1) = " << outputLayer.w[0](i, j) << "\n";
//    cout << "finite diff: " << (Oplus - Ominus)/(2*epsilon) << "\n";
    cout << "w_ic(1,1) = " << forwardHiddenLayer.hiddenLayerNodes[c].w_ic(i) << "\n";
    forwardHiddenLayer.hiddenLayerNodes[c].w_iig(i) += epsilon;
    forwardHiddenLayer.forwardPass(inputs); //for each input sequence (image with a word)
    backwardHiddenLayer.forwardPass(inputs);
    outputLayer.forwardPass(inputs.size(), label, forwardHiddenLayer.b_c, backwardHiddenLayer.b_c);
    outputLayer.backwardPass();
    double Ominus = outputLayer.computeObjectiveFunction();
    cout << "w_ic(1,1) = " << forwardHiddenLayer.hiddenLayerNodes[c].w_iig(i) << "\n";

    forwardHiddenLayer.hiddenLayerNodes[c].w_iig(i) -= 2*epsilon; //because I need to subtract the previous addition
    forwardHiddenLayer.forwardPass(inputs); //for each input sequence (image with a word)
    backwardHiddenLayer.forwardPass(inputs);
    outputLayer.forwardPass(inputs.size(), label, forwardHiddenLayer.b_c, backwardHiddenLayer.b_c);
    outputLayer.backwardPass();
    double Oplus = outputLayer.computeObjectiveFunction();
    cout << "w_ic(1,1) = " << forwardHiddenLayer.hiddenLayerNodes[c].w_iig(i) << "\n";
    cout << "finite diff: " << (Oplus - Ominus)/(2*epsilon) << "\n";

}

/**
 * validate network
 */
void NeuralNetwork::evaluateValidationSet(vector<string> validationset, ImagesHandler im_handler) {

    cout << "VALIDATE!\n";
    int setSize = 100;///validationset.size();

    for(int i = 0; i < setSize; ++ i)
    {
        string imagePath(validationset[i]);
        FeatureExtractor extractor(imagePath);
        vector< VectorXd > sequenceOfFeatures = extractor.getFeatures();
        string label = im_handler.getTargetLabel(imagePath);
//        cout << imagePath << "; label =" << label << "=" << endl;
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

