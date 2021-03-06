#include "neuralnetwork.h"

void NeuralNetwork::trainNetwork(int off) {

    int  stepSize = 100; //1000
    int offset = stepSize*off;
    int nbExamples;
    ImagesHandler im_handler;
    vector<string> trainset = im_handler.getDataSet("trainset.txt");
    vector<string> validationset = im_handler.getDataSet("validationset1.txt");
    nbExamples = 100;//trainset.size();
    double prevValidationError = 10e5;

    cout << "trainset_size: " << trainset.size() << "\n";
    cout << "validationset_size: " << validationset.size() << "\n";

    out.open("trainingErrors100Eg_100ITER50LSTM.001ETA+MIU+Valid.txt");

    if (!out.is_open())
    {
        cout << "Failed to open file...\n";
        return;
    }

    //in a loop -- train the weights until a stop condition is fullfilled
    for(int epoch = 0; epoch < 100/*MAX_ITER*/; ++ epoch)
    {

        outputLayer.trainError = 0.0; // the training error
        outputLayer.validationError = 0.0;

        random_shuffle ( trainset.begin() + offset, trainset.begin() + offset + nbExamples );

        for(int eg = offset; eg < offset + nbExamples && eg < trainset.size(); ++ eg)
        {
            string imagePath(trainset[eg]);
            FeatureExtractor extractor(imagePath);
            vector< VectorXd > sequenceOfFeatures = extractor.getFeatures();
            string label = im_handler.getTargetLabel(imagePath);
//            cout << imagePath << "; label =" << label << "=" << endl;
            trainOneExample( sequenceOfFeatures, label );
//            trainOneExampleDebug( sequenceOfFeatures, label );
        }
        outputLayer.trainError /= nbExamples;
        cout << "ctcError: " << outputLayer.trainError << "\n";
        cout << "*********************************************************\n";
        evaluateValidationSet(validationset, im_handler, off);
        out << (epoch+1) << " " << outputLayer.trainError << " ";
        out << outputLayer.validationError << "\n";
        if(epoch > 20 && outputLayer.validationError - prevValidationError > 0.1) //error starts to increase
            break;
        prevValidationError = outputLayer.validationError;//min(minValidationError, outputLayer.validationError);

    }

    out.close();

    //save network weights
    ofstream printout;
    printout.open(NETWORK_WEIGHTS_FILENAME);

    if (!printout.is_open())
    {
        cout << "Failed to open file "  << NETWORK_WEIGHTS_FILENAME << "\n";
        return;
    }

    printout << I << " " << H << " " << K << "\n";
    forwardHiddenLayer.printWeights(printout);
    backwardHiddenLayer.printWeights(printout);
    outputLayer.printWeights(printout);
    printout.close();

}

void NeuralNetwork::trainNetworkDebug(string imagePath) {

    int nbExamples;
    ImagesHandler im_handler;
    vector<string> trainset = im_handler.getDataSet("trainset.txt");
//    vector<string> validationset = im_handler.getDataSet("validationset1.txt");
    nbExamples = 10;//trainset.size();

//    cout << "trainset_size: " << trainset.size() << "\n";
//    cout << "validationset_size: " << validationset.size() << "\n";

//    out.open("trainErrors10Eg_10Iter10LSTM.01ETA+MIU+Valid.txt");

//    if (!out.is_open())
//    {
//        cout << "Failed to open file...\n";
//        return;
//    }

    //in a loop -- train the weights until a stop condition is fullfilled
    for(int epoch = 0; epoch < 1; ++ epoch)
    {

        outputLayer.trainError = 0.0; // the training error
        outputLayer.validationError = 0.0;
//        for(int eg = offset; eg < offset + nbExamples && eg < trainset.size(); ++eg)
//        {
            int eg = 0;
            string imagePath(trainset[eg]);
            FeatureExtractor extractor(imagePath);
            vector< VectorXd > sequenceOfFeatures = extractor.getFeatures();
//            vector<VectorXd> sequenceOfFeatures;
//            sequenceOfFeatures.push_back(originalSequenceOfFeatures[0]);
//            sequenceOfFeatures.push_back(originalSequenceOfFeatures[1]);
            string label = im_handler.getTargetLabel(imagePath);
            cout << imagePath << "; label =" << label << "=" << endl;
            trainOneExampleDebug( sequenceOfFeatures, label );
//            trainOneExampleWithPrints( sequenceOfFeatures, label );
//        }
        outputLayer.trainError /= nbExamples;
        cout << "ctcError: " << outputLayer.trainError << "\n";
        cout << "*********************************************************\n";

    }

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

void NeuralNetwork::trainOneExampleWithPrints(vector<VectorXd> x, string label) {

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
//    vector<MatrixXd> eps_c1 = outputLayer.getEpsilonCTC();
//    forwardHiddenLayer.backwardPass(eps_c1[0]);
//    backwardHiddenLayer.backwardPass(eps_c1[1]);

//    /// update weights
//    outputLayer.updateWeights(ETA);
//    forwardHiddenLayer.updateWeights(ETA);
//    backwardHiddenLayer.updateWeights(ETA);

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
    //DEBUG - gradient!
    double epsilon = 1e-5;
    int c = 0, i = 0, j = 0;

    /// backward pass
    outputLayer.backwardPass();
    vector<MatrixXd> eps_c1 = outputLayer.getEpsilonCTC();
    forwardHiddenLayer.backwardPass(eps_c1[0]);
    backwardHiddenLayer.backwardPass(eps_c1[1]);

    /// update weights
    outputLayer.updateWeights(ETA);
    forwardHiddenLayer.updateWeights(ETA);
//    backwardHiddenLayer.updateWeights(ETA);


//    cout << "AFTER passing -- one example\n";
    cout << "#############################################################\n";


//    cout << "w[0](1,1) = " << outputLayer.w[0](i,j) << "\n";
//    outputLayer.w[0](i, j) += epsilon;
//    outputLayer.forwardPass(inputs.size(), label, forwardHiddenLayer.b_c, backwardHiddenLayer.b_c);
//    outputLayer.backwardPass();
//    double Oplus = -outputLayer.computeObjectiveFunction();
//    cout << "w[0](1,1) = " << outputLayer.w[0](i,j) << "\n";

//    outputLayer.w[0](i, j) -= 2*epsilon; //because I need to subtract the previous addition
//    outputLayer.forwardPass(inputs.size(), label, forwardHiddenLayer.b_c, backwardHiddenLayer.b_c);
//    outputLayer.backwardPass();
//    double Ominus = -outputLayer.computeObjectiveFunction();
//    cout << "w[0](1,1) = " << outputLayer.w[0](i, j) << "\n";
//    cout << "finite diff: " << (Oplus - Ominus)/(2*epsilon) << "\n";

    cout << "[NN] before:w_hig(1,1) = " << forwardHiddenLayer.hiddenLayerNodes[c].w_hig(i) << "\n";
    forwardHiddenLayer.hiddenLayerNodes[c].w_hig(i) += epsilon;
    forwardHiddenLayer.forwardPass(inputs); //for each input sequence (image with a word)
    backwardHiddenLayer.forwardPass(inputs);
    outputLayer.forwardPass(inputs.size(), label, forwardHiddenLayer.b_c, backwardHiddenLayer.b_c);
    outputLayer.backwardPass();
    double Oplus = -outputLayer.computeObjectiveFunction();
    cout << "[NN]w_hig(1,1) = " << forwardHiddenLayer.hiddenLayerNodes[c].w_hig(i) << "\n";

    forwardHiddenLayer.hiddenLayerNodes[c].w_hig(i) -= 2*epsilon; //because I need to subtract the previous addition
    forwardHiddenLayer.forwardPass(inputs); //for each input sequence (image with a word)
    backwardHiddenLayer.forwardPass(inputs);
    outputLayer.forwardPass(inputs.size(), label, forwardHiddenLayer.b_c, backwardHiddenLayer.b_c);
    outputLayer.backwardPass();
    double Ominus = -outputLayer.computeObjectiveFunction();
    cout << "[NN]w_hig(1,1) = " << forwardHiddenLayer.hiddenLayerNodes[c].w_hig(i) << "\n";
    cout << "finite diff: " << (Oplus - Ominus)/(2*epsilon) << "\n";

}

/**
 * validate network
 */
void NeuralNetwork::evaluateValidationSet(vector<string> validationset, ImagesHandler im_handler, int off) {

    cout << "VALIDATE!\n";
    int setSize = 50;///validationset.size();
    int stepSize = 100;//500
    int offset = off*stepSize;

    for(int i = offset; i < offset + setSize; ++ i)
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

    outputLayer.validationError /= setSize;

}

//test network
void NeuralNetwork::testInputImage(string imagePath, ImagesHandler im_handler) {

//    cout << "TEST!\n";

    FeatureExtractor extractor(imagePath);
    vector< VectorXd > sequenceOfFeatures = extractor.getFeatures();
    string label = im_handler.getTargetLabel(imagePath);
//    cout << imagePath << "; label =" << label << "=" << endl;
    inputs = sequenceOfFeatures;

    /// forward pass
    forwardHiddenLayer.forwardPass(inputs);
    backwardHiddenLayer.forwardPass(inputs);
    outputLayer.forwardPass(inputs.size(), label, forwardHiddenLayer.b_c, backwardHiddenLayer.b_c);

}


