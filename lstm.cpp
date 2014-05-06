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
    this->C = 1;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine g(seed);
    std::normal_distribution<double> distrib (0.0, 0.1);
    distribution = distrib;
    generator = g;
    if(DEBUG)
        cout << "Constructor LSTM\n";
    initWeights();
    initDeltaWeights();
}

LSTM::LSTM(int inputUnitsNum, int hiddenUnitsNum, istream& fin) {
    this->I= inputUnitsNum;
    this->H = hiddenUnitsNum;
    this->C = 1;
    readWeights(fin);
    initDeltaWeights();
}


LSTM::~LSTM() {
    // TODO Auto-generated destructor stub
}

void LSTM::initWeights() {

    w_iig = VectorXd::Zero(I);//initRandomVector(I); //cout << "W_IIG " << w_iig(0) << "\n";
    w_ifg = VectorXd::Zero(I);//initRandomVector(I); //VectorXd::Zero(I);
    w_iog = VectorXd::Zero(I);//initRandomVector(I);

    //recurrent connections
    w_hig = VectorXd::Zero(H);//initRandomVector(H);
    w_hfg = VectorXd::Zero(H);//initRandomVector(H);
    w_hog = VectorXd::Zero(H);//initRandomVector(H); //correct!

    //peephole connections (between cells inside an LSTM
    w_cig = VectorXd::Zero(C);//initRandomVector(C);
    w_cfg = VectorXd::Zero(C);//initRandomVector(C);
    w_cog = VectorXd::Zero(C);//initRandomVector(C); //correct!

    w_ic = initRandomVector(I);
    w_hc = VectorXd::Zero(H);//initRandomVector(H); //
}

void LSTM::readWeights(istream &fin) {
    readVector(w_iig, I, fin);
    readVector(w_ifg, I, fin);
    readVector(w_iog, I, fin);

    readVector(w_hig, H, fin);
    readVector(w_hfg, H, fin);
    readVector(w_hog, H, fin);

    readVector(w_cig, C, fin);
    readVector(w_cfg, C, fin);
    readVector(w_cog, C, fin);

    readVector(w_ic, I, fin);
    readVector(w_hc, H, fin);
}


void LSTM::printWeights(ostream &out) {
    printVector(w_iig, I, out);
    printVector(w_ifg, I, out);
    printVector(w_iog, I, out);

    printVector(w_hig, H, out);
    printVector(w_hfg, H, out);
    printVector(w_hog, H, out);

    printVector(w_cig, C, out);
    printVector(w_cfg, C, out);
    printVector(w_cog, C, out);

    printVector(w_ic, I, out);
    printVector(w_hc, H, out);
}

void LSTM::initDeltaWeights() {

    delta_w_iig = VectorXd::Zero(I);
    delta_w_ifg = VectorXd::Zero(I);
    delta_w_iog = VectorXd::Zero(I);

    delta_w_hig = VectorXd::Zero(H);
    delta_w_hfg = VectorXd::Zero(H);
    delta_w_hog = VectorXd::Zero(H);

    delta_w_cig = VectorXd::Zero(C);
    delta_w_cfg = VectorXd::Zero(C);
    delta_w_cog = VectorXd::Zero(C);

    delta_w_ic = VectorXd::Zero(I);
    delta_w_hc = VectorXd::Zero(H);

}

void LSTM::startNewForwardPass(VectorXd x, VectorXd p_b, VectorXd p_sc)
{
    this->x_t = x;
    this->prev_b = p_b;
    this->prev_sc = p_sc;
}

double LSTM::forwardPassInputGate() {
    if (DEBUG) std::cout << "INPUT-GATE" << "\n";
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

void LSTM::printVector(VectorXd w, int size, ostream& out) {
    for (int i = 0; i < size; ++i) {
        out << w[i] << " ";
    }
    out << "\n";

}

void LSTM::readVector(VectorXd& w, int size, istream& fin) {
    w = VectorXd::Zero(size);
    for (int i = 0; i < size; ++i) {
        fin >> w[i];
    }
}
