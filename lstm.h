/*
 * LSTM.h
 *
 *  Created on: Mar 16, 2014
 *      Author: silvia
 */

#ifndef LSTM_H_
#define LSTM_H_
#include <iostream>
#include <chrono>
#include <random>
#include <time.h>
#include "../Eigen/Core"
#include "utils.h"

using namespace Eigen;
using namespace std;

/**
 * A class that implements the logic behind a long short-term memory cell
 */
class LSTM {
    int I; //#input units
    int H; //#hidden units;
    int C; //C = H - cells inside a LSTM block
    VectorXd x_t; // input at time t
    VectorXd prev_b; // output (activations) at timestep t-1
    VectorXd prev_sc; //cell states at previous timestep: t-1

    std::default_random_engine generator;
    std::normal_distribution<double> distribution;
public:
    VectorXd w_iig, w_ifg, w_iog; //weights between inputs and gates
    VectorXd w_cig, w_cfg, w_cog; //weights between cell-states and gates
    VectorXd w_hig, w_hfg, w_hog; //weights between hidden units and gates
    VectorXd w_ic, w_hc; //weights between cells and input/hidden units
    //delta weights from the previous epoch used to update the weights using momentum term
    VectorXd delta_w_iig, delta_w_ifg, delta_w_iog; //weights between inputs and gates
    VectorXd delta_w_cig, delta_w_cfg, delta_w_cog; //weights between cell-states and gates
    VectorXd delta_w_hig, delta_w_hfg, delta_w_hog; //weights between hidden units and gates
    VectorXd delta_w_ic, delta_w_hc; //weights between cells and input/hidden units

    LSTM(int inputUnitsNum, int hiddenUnitsNum);
    virtual ~LSTM();
    void initWeights();
    void iniDeltaWeights();

    void startNewForwardPass(VectorXd x, VectorXd b, VectorXd sc);
    double forwardPassInputGate();
    double forwardPassForgetGate();
    double forwardPassOutputGate(VectorXd sc_t);
    double forwardPassCell();

private:
    VectorXd initRandomVector(int size);
};

#endif /* LSTM_H_ */
