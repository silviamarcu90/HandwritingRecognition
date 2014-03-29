/*
 * LSTM.cpp
 *
 *  Created on: Mar 16, 2014
 *      Author: silvia
 */

#include "lstm.h"
#include <stdio.h>

LSTM::LSTM(int inputUnitsNum, int hiddenUnitsNum) {
    this->I= inputUnitsNum;
    this->H = hiddenUnitsNum;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine g(seed);
    std::normal_distribution<double> distrib (0.0, 0.1);
    distribution = distrib;
    generator = g;
    initWeights();
}

LSTM::~LSTM() {
    // TODO Auto-generated destructor stub
}

void LSTM::initWeights() {

    w_iig = initRandomVector(I);
    w_ifg = initRandomVector(I);
    w_iog = initRandomVector(I);

    w_hig = initRandomVector(H);
    w_hfg = initRandomVector(H);
    w_hog = initRandomVector(H);

    w_cig = initRandomVector(H);
    w_cfg = initRandomVector(H);
    w_cog = initRandomVector(H);

    w_ic = initRandomVector(I);
    w_hc = initRandomVector(H);
}

void LSTM::startNewForwardPass(VectorXd x, VectorXd p_b, VectorXd p_sc)
{
    this->x_t = x;
    this->prev_b = p_b;
    this->prev_sc = p_sc;
}

double LSTM::forwardPassInputGate() {
    if (DEBUG)
        std::cout << "INPUT-GATE" << "\n";
    double a = w_iig.dot(x_t) + w_hig.dot(prev_b) + w_cig.dot(prev_sc);
//    std::cout << "aa" << ":" << a << " ";
//    printf("in-rez: %lf\n", a);
    return a;
}

double LSTM::forwardPassForgetGate() {
    if (DEBUG) std::cout << "FORGET-GATE" << "\n";
    return w_ifg.dot(x_t) + w_hfg.dot(prev_b) + w_cfg.dot(prev_sc);
}

double LSTM::forwardPassOutputGate(VectorXd sc_t) {
    if (DEBUG) std::cout << "OUTPUT-GATE " << sc_t.size() << " \n";
//    std::cout << "weights sizes "<< w_iog.size() << " " << w_hog.size() << " " << w_cog.size() << "\n";
//    std::cout << "other elems " << x_t.size() << ", " << prev_b.size() << "; " << sc_t.size() << "\n";
    return w_iog.dot(x_t) + w_hog.dot(prev_b) + w_cog.dot(sc_t);
}

double LSTM::forwardPassCell() {
    if (DEBUG) std::cout << "FORWARD -- pass cell\n";
    return w_ic.dot(x_t) + w_hc.dot(prev_b);
}

//void  LSTM::backwardPassCellOutput(vector<MatrixXd> eps_c1) {

//}

VectorXd LSTM::initRandomVector(int size) {
    VectorXd w;
    w = VectorXd::Zero(size);
//    std::cout << "ww ";
    for (int i = 0; i < size; ++i) {
        w[i] = distribution(generator);
//        std::cout << " " << w[i] << " ";
    }
//    std::cout << std::endl;
    return w;
}

