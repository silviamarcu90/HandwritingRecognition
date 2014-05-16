#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <fstream>
#include <algorithm>
#include "blstm.h"
#include "forwardlayerlstm.h"
#include "backwardlayerlstm.h"
#include "ctclayer.h"
#include "featureextractor.h"
#include "imageshandler.h"

#define MAX_ITER 10 /// maximum number of iterations(epochs) -- used to stop training
#define NETWORK_WEIGHTS_FILENAME "NETWORK_WEIGHTS"

class NeuralNetwork
{
    double ETA; /// the learning rate
    int I, H, K;
    vector<VectorXd> inputs; /// vector of T elements, each element with 9 features
    ForwardLayerLSTM forwardHiddenLayer; /// the hidden layer that traverse the input sequence in forward order
    BackwardLayerLSTM backwardHiddenLayer; /// traverse the input sequence in the reverse order

public:
    CTCLayer outputLayer; /// connectionist temporal classification -- output layer
    ofstream out;

    NeuralNetwork(int hiddenUnitsNum, int outputUnitsNum) :
        I(NB_FEATURES),
        H(hiddenUnitsNum),
        K(outputUnitsNum),
        ETA(0.001),
        forwardHiddenLayer(hiddenUnitsNum),
        backwardHiddenLayer(hiddenUnitsNum),
        outputLayer(outputUnitsNum, hiddenUnitsNum) {
        cout << "Constructor - NeuralNetwork\n";
    }

    /**
     * Constructor used to recover the network weights trained before
     */
    NeuralNetwork(string netWeightsFile) : forwardHiddenLayer(0),
        backwardHiddenLayer(0) {

        ifstream fin(netWeightsFile);
        if (!fin.is_open())
        {
            cout << "Failed to open file... " << netWeightsFile << " \n";
            return;
        }

        fin >> I >> H >> K;

        ETA = 0.01;
        ForwardLayerLSTM l1(H, fin);
        forwardHiddenLayer = l1;
        BackwardLayerLSTM l2(H, fin);
        backwardHiddenLayer = l2;
        CTCLayer l3(K, H, fin);
        outputLayer = l3;
        fin.close();

        //check
        if(DEBUG) {
            ofstream printout;
            printout.open("NETWORK_WEIGHTS_TEST");

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
    }

    void trainNetwork(int offset);
    void trainNetworkDebug(string imagePath);

    void trainOneExample(vector<VectorXd> x, string label);
    void trainOneExampleDebug(vector<VectorXd> x, string label);
    void trainOneExampleWithPrints(vector<VectorXd> x, string label);

    void evaluateValidationSet(vector<string> validationset, ImagesHandler im_handler, int off);

    void testInputImage(string imagePath, ImagesHandler im_handler);

};

#endif // NEURALNETWORK_H
