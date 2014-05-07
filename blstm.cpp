/*
 * BLSTM.cpp
 *
 *  Created on: Mar 16, 2014
 *      Author: silvia
 */

#include "blstm.h"

//BLSTM::BLSTM() {

//}

BLSTM::BLSTM(int hiddenUnitsNum) {
    H = hiddenUnitsNum;
    this->I = NB_FEATURES; //9 features for the input layer
    for(int i = 0; i < H; ++i)
    {
        LSTM unit(I, H);
        hiddenLayerNodes.push_back(unit);
    }

}

BLSTM::BLSTM(int hiddenUnitsNum, istream& fin) {
    H = hiddenUnitsNum;
    this->I = NB_FEATURES; //9 features for the input layer
    for(int i = 0; i < H; ++i)
    {
        LSTM unit(I, H, fin);
        hiddenLayerNodes.push_back(unit);
    }

}

BLSTM::~BLSTM() {    
    // TODO Auto-generated destructor stub
}

void BLSTM::initActivationsAndDelta(vector< VectorXd > input) {
//    this->I = input[0].size();
    this->T = input.size();
    this->C = 1;
    x = input;
    b_c = reserve(T, H);
    sc = reserve(T, H);
    a_c = reserve(T, H);
    a_og = reserve(T, H);
    b_og = reserve(T, H);
    a_fg = reserve(T, H);
    b_fg = reserve(T, H);
    a_ig = reserve(T, H);
    b_ig = reserve(T, H);

    delta_o = MatrixXd::Zero(T + 1, H);
    delta_c = MatrixXd::Zero(T + 1, H);
    delta_f = MatrixXd::Zero(T + 1, H);
    delta_i = MatrixXd::Zero(T + 1, H);
    if(DEBUG) std::cout << "BLSTM --- init LSTM-cells\n";
//    for(int i = 0; i < H; ++i)
//    {
//        LSTM unit(I, H);
//        hiddenLayerNodes.push_back(unit);
//    }

}

void BLSTM::printWeights(ostream &out) {
    for(int i = 0; i < H; ++i)
    {
        hiddenLayerNodes[i].printWeights(out);
        out << "\n";
    }

}

void BLSTM::forwardPass(vector< VectorXd > input) {

}

void BLSTM::backwardPass(MatrixXd eps_c1) {

}

void BLSTM::updateWeightsOfCellInputGate(int c) {}
void BLSTM::updateWeightsOfCellForgetGate(int c) {}
void BLSTM::updateWeightsOfCellOutputGate(int c) {}
void BLSTM::updateWeightsOfCellState(int c) {}
void BLSTM::updateWeights(double ETA) {}

void BLSTM::print() {
    cout << "b_c[t](i): \n";
    for(int t = 0; t < T; ++t)
    for(int i = 0; i < H; ++i)
        cout << b_c[t](i) << " ";
    cout << "\n";
}

vector<VectorXd> BLSTM::reserve(int s1, int s2) {
    vector<VectorXd> vec;
    for(int i = 0; i < s1; ++i) {
        VectorXd elem = VectorXd::Zero(s2);
        vec.push_back(elem);
    }
    return vec;
}
