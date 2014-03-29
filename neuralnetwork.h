#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "blstm.h"
#include "forwardlayerlstm.h"
#include "backwardlayerlstm.h"
#include "ctclayer.h"
#include "featureextractor.h"

class NeuralNetwork
{
    double ETA; /// the learning rate
    vector<VectorXd> inputs; /// vector of T elements, each element with 9 features
    ForwardLayerLSTM forwardHiddenLayer; /// the hidden layer that traverse the input sequence in forward order
    BackwardLayerLSTM backwardHiddenLayer; /// traverse the input sequence in the reverse order
    CTCLayer outputLayer; /// connectionist temporal classification -- output layer

public:
    NeuralNetwork(int hiddenUnitsNum, int outputUnitsNum) :
        forwardHiddenLayer(hiddenUnitsNum),
        backwardHiddenLayer(hiddenUnitsNum),
        outputLayer(outputUnitsNum, hiddenUnitsNum) {
         cout << "Constructor - NeuralNetwork\n";
    }

    void trainNetwork();

    void trainExample(vector<VectorXd> x, string label);

};

#endif // NEURALNETWORK_H
