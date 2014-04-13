#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <fstream>
#include "blstm.h"
#include "forwardlayerlstm.h"
#include "backwardlayerlstm.h"
#include "ctclayer.h"
#include "featureextractor.h"
#include "imageshandler.h"

#define MAX_ITER 10 /// maximum number of iterations(epochs) -- used to stop training

class NeuralNetwork
{
    double ETA; /// the learning rate
    vector<VectorXd> inputs; /// vector of T elements, each element with 9 features
    ForwardLayerLSTM forwardHiddenLayer; /// the hidden layer that traverse the input sequence in forward order
    BackwardLayerLSTM backwardHiddenLayer; /// traverse the input sequence in the reverse order
    CTCLayer outputLayer; /// connectionist temporal classification -- output layer

public:
    ofstream out;
    NeuralNetwork(int hiddenUnitsNum, int outputUnitsNum) :
        ETA(0.001),
        forwardHiddenLayer(hiddenUnitsNum),
        backwardHiddenLayer(hiddenUnitsNum),
        outputLayer(outputUnitsNum, hiddenUnitsNum) {
        cout << "Constructor - NeuralNetwork\n";
    }

    void trainNetwork();

    void trainOneExample(vector<VectorXd> x, string label);

    void evaluateValidationSet(vector<string> validationset, ImagesHandler im_handler);


};

#endif // NEURALNETWORK_H
