/*
 * BLSTM.h
 *
 *  Created on: Mar 16, 2014
 *      Author: silvia
 */

#ifndef BLSTM_H_
#define BLSTM_H_

#include <iostream>
#include <vector>
#include "lstm.h"
#include "utils.h"

/**
 * Class that implements the Bidirectional Long Short Term Memory
 */
class BLSTM {
    int H; /// number of hidden units
    int T; /// number of terms in the input sequence
    int direction; /// 1 = forward, 2 = backward
    vector<VectorXd> x; /// input sequence of length T
    //VectorXd prev_b; /// output (activations) at timestep t-1
    vector<LSTM> hiddenLayerNodes;
    vector<VectorXd> b_c; /// output of memory blocks (LSTM) at each time t -- for the forward sequence
    vector<VectorXd> a_og, b_og; //activations of the output gate
    vector<VectorXd> sc;
    vector<VectorXd> a_c;
    vector<VectorXd> a_fg, b_fg; //activations of the forget gate
    vector<VectorXd> a_ig, b_ig; //activations of the input gate
    /// residuals used to update weights in the backward pass
    MatrixXd delta_o; /// for the output gate
    MatrixXd delta_c; /// for the cell
    MatrixXd delta_f; /// for the forget gate
    MatrixXd delta_i; /// for the input gate
public:
    BLSTM(int hiddenUnitsNum, vector<VectorXd> x);
    virtual ~BLSTM();
    void forwardPass();
    void backwardPass(MatrixXd eps_c1);
    vector<VectorXd> reserve(int s1, int s2);

};

#endif /* BLSTM_H_ */
