#ifndef CTCLAYER_H
#define CTCLAYER_H

#include <iostream>
#include <chrono>
#include <random>
#include <string.h>
#include <vector>
#include <map>
#include "float.h"
#include "../Eigen/Core"
#include "utils.h"

using namespace Eigen;
using namespace std;

class CTCLayer {
    int H; /// number of hidden units
    int K; /// number of output units
    int T; /// input sequence length
    string l, l_prime;
    vector< MatrixXd > w; /// weights for the computation of the output units: T x H x K
    MatrixXd a, y; /// activations and softmax function results: T x K
    MatrixXd alpha, beta; /// forward/backward variables: T x K
    MatrixXd delta_k; /// residual variables of the CTC layer: T x K
    vector<VectorXd> forward_b, backward_b; /// inputs of this layer (outputs of the BLSTM hidden layer)
    map<char, int> alphabet;
    vector<double> cond_probabs; /// p(z|x) at each timestep t (combination between alpha and beta)
    vector<MatrixXd> eps_c1; /// variable necessary for the backpropagation in the hidden layer: 2 elements [forward; backward]

public:
    CTCLayer();
    CTCLayer(int K, int H);
    virtual ~CTCLayer();
    void forwardPass(int T, string label, vector<VectorXd> forward_b, vector<VectorXd> backward_b);
    void backwardPass();
    void computeForwardVariable();
    void computeBackwardVariable();
    void updateWeights(double ETA);
    vector<MatrixXd> getEpsilonCTC();

private:
    void initWeights();
    void initActivations();
    MatrixXd initRandomMatrix(int m, int n);
    MatrixXd compute_exp(MatrixXd a);
    void initAlphabet();
    char getKeyByValue(int k); /// get the key in alphabet-map knowing the value
    void createExtendedLabel();
    double f_u(int u);
    double g_u(int u);
    double computeProbability(int k, int t);

};

#endif // CTCLAYER_H
