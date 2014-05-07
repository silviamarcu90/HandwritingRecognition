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

    w_iig = initRandomVector(I);//VectorXd::Zero(I);// //cout << "W_IIG " << w_iig(0) << "\n";
    w_ifg = initRandomVector(I); //VectorXd::Zero(I);
    w_iog = initRandomVector(I);

    //recurrent connections
    w_hig = initRandomVector(H);//VectorXd::Zero(H);//
    w_hfg = initRandomVector(H); //VectorXd::Zero(H);//initRandomVector(H);
    w_hog = initRandomVector(H);//initRandomVector(H); //correct!

    //peephole connections (between cell inside an LSTM and the gates)
    w_cig = initRandomVector(C);
    w_cfg = initRandomVector(C);
    w_cog = initRandomVector(C); //correct!

    w_ic = initRandomVector(I);
    w_hc = initRandomVector(H); //VectorXd::Zero(H);//


//    w_iig = initRandomVector(I);//VectorXd::Zero(I);// //cout << "W_IIG " << w_iig(0) << "\n";
//    w_ifg = initRandomVector(I); //VectorXd::Zero(I);
//    w_iog = initRandomVector(I);
////    w_iig = initConstantVector(I, 0.15);//VectorXd::Zero(I);//initRandomVector(I); //cout << "W_IIG " << w_iig(0) << "\n";
////    w_ifg = initConstantVector(I, 0.05);//initRandomVector(I); //VectorXd::Zero(I);
////    w_iog = initConstantVector(I, -0.25);//initRandomVector(I);

//    //recurrent connections
//    w_hig = initRandomVector(H);//VectorXd::Zero(H);//
//    w_hfg = initRandomVector(H); //VectorXd::Zero(H);//initRandomVector(H);
//    w_hog = initRandomVector(H);//initRandomVector(H); //correct!

////    w_hig = initConstantVector(H, 0.3);//VectorXd::Zero(H);//initRandomVector(H);
////    w_hig(1) = 0.4;
////    w_hfg = initConstantVector(H, -0.1); //VectorXd::Zero(H);//initRandomVector(H);
////    w_hog = initConstantVector(H, -0.1);//initRandomVector(H); //correct!

//    //peephole connections (between cells inside an LSTM
//    w_cig = initRandomVector(C);
//    w_cfg = initRandomVector(C);
//    w_cog = initRandomVector(C); //correct!
////    w_cig = initConstantVector(C, 0.03);//initRandomVector(C);
////    w_cfg = initConstantVector(C, 0.03);//initRandomVector(C);
////    w_cog = initConstantVector(C, 0.03);//initRandomVector(C); //correct!

//    w_ic = initRandomVector(I);
//    w_hc = initRandomVector(H); //VectorXd::Zero(H);//

//    w_ic = initConstantVector(I, 0.05);
//    w_hc = initConstantVector(H, 0.1);//VectorXd::Zero(H);//initRandomVector(H); //
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
    double a = w_iig.dot(x_t) + w_hig.dot(prev_b) + w_cig.dot(prev_sc);
    if (DEBUG) std::cout << "INPUT-GATE - a_ig = " << a << "\n";
//    std::cout << "aa" << ":" << a << " ";
//    printf("in-rez: %lf\n", a);
    return a;
}

double LSTM::forwardPassForgetGate() {
    double a = w_ifg.dot(x_t) + w_hfg.dot(prev_b) + w_cfg.dot(prev_sc);
    if (DEBUG) std::cout << "FORGET-GATE a_fg " << a << "\n";
    return a;
}

double LSTM::forwardPassOutputGate(VectorXd sc_t) {
//    std::cout << "weights sizes "<< w_iog.size() << " " << w_hog.size() << " " << w_cog.size() << "\n";
//    std::cout << "other elems " << x_t.size() << ", " << prev_b.size() << "; " << sc_t.size() << "\n";
    double a = w_iog.dot(x_t) + w_hog.dot(prev_b) + w_cog.dot(sc_t);
    if (DEBUG) std::cout << "OUTPUT-GATE a_og " << a << " \n";
    return a;
}

double LSTM::forwardPassCell() {
    double a = w_ic.dot(x_t) + w_hc.dot(prev_b);
    if (DEBUG) std::cout << "FORWARD -- pass cell a_c = " << a << "\n";
    return a;
}


VectorXd LSTM::initRandomVector(int size) {
    VectorXd w;
    w = VectorXd::Zero(size);
    for (int i = 0; i < size; ++i) {
        w[i] = distribution(generator);
//        std::cout << " " << w[i] << " ";
    }
//    std::cout << std::endl;
    return w;
}

VectorXd LSTM::initConstantVector(int size, double ct) {
    VectorXd w;
    w = VectorXd::Zero(size);
    for (int i = 0; i < size; ++i) {
        w[i] = ct;
    }
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
