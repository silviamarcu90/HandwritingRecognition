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
}

BLSTM::~BLSTM() {
    // TODO Auto-generated destructor stub
}

void BLSTM::initActivationsAndDelta(vector< VectorXd > input) {
    this->I = input[0].size();
    this->T = input.size();
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
//    MatrixXd delta_aux(T + 1, H);
    delta_o = MatrixXd::Zero(T + 1, H);
    delta_c = MatrixXd::Zero(T + 1, H);
    delta_f = MatrixXd::Zero(T + 1, H);
    delta_i = MatrixXd::Zero(T + 1, H);
    std::cout << "BLSTM --- init LSTM-cells\n";
    for(int i = 0; i < H; ++i)
    {
        LSTM unit(I, H);
        hiddenLayerNodes.push_back(unit);
    }

}

void BLSTM::forwardPass(vector< VectorXd > input) {

}

void BLSTM::backwardPass(MatrixXd eps_c1) {

}

void BLSTM::updateWeights(double eta) {
    int c;

    ETA = eta;
    //for each cell in the hidden layer
    for(c = 0; c < H; ++c) {
        updateWeightsOfCellInputGate(c);
        updateWeightsOfCellForgetGate(c);
        updateWeightsOfCellOutputGate(c);
        updateWeightsOfCellState(c);
    }

}

//!!!!!!!!!check inputs that enter in a cell -- if they are properly used for updating!!! -- time t or t-1 -- ask??

void BLSTM::updateWeightsOfCellState(int c) {

    for(int i = 0; i < I; ++i) {
        double gradient_i = 0; //the gradient of the inputs-input_gate weights
        for(int t = 0; t < T; ++t) {
            gradient_i += delta_c(t, c)*x[t](i);
        }
        hiddenLayerNodes[c].w_ic(i) -= ETA*gradient_i;
    }

    for(int i = 0; i < H; ++i) {
        double gradient_h = 0; //the gradient of the hidden_units-input_gate weights
        for(int t = 0; t < T; ++t) {
            gradient_h += delta_c(t, c)*b_c[t](i);
        }
        hiddenLayerNodes[c].w_hc(i) -= ETA*gradient_h;
    }

}

void BLSTM::updateWeightsOfCellInputGate(int c) {

    for(int i = 0; i < I; ++i) {
        double gradient_i = 0; //the gradient of the inputs-input_gate weights
        for(int t = 0; t < T; ++t) {
            gradient_i += delta_i(t, c)*x[t](i);
        }
        hiddenLayerNodes[c].w_iig(i) -= ETA*gradient_i;
//        cout << "w_iig " << hiddenLayerNodes[c].w_iig(i) << " ";
    }
//    cout << "\n";

    for(int i = 0; i < H; ++i) {
        double gradient_h = 0; //the gradient of the hidden_units-input_gate weights
        for(int t = 0; t < T; ++t) {
            gradient_h += delta_i(t, c)*b_c[t](i);
        }
        hiddenLayerNodes[c].w_hig(i) -= ETA*gradient_h;
//        cout << "w_hig " << hiddenLayerNodes[c].w_hig(i) << " ";
    }
//    cout << "\n";

    for(int i = 0; i < H; ++i) {
        double gradient_c = 0; //the gradient of the cell_states - input_gate weights
        for(int t = 0; t < T; ++t) {
            gradient_c += delta_i(t, c)*sc[t](i); /// assumption -- the input of this node is the output of interior cell s_c
        }
        hiddenLayerNodes[c].w_cig(i) -= ETA*gradient_c;
//        cout << "w_cig " << hiddenLayerNodes[c].w_cig(i) << " "; //!? too big values (sometimes)
    }
//    cout << "\n";

}


void BLSTM::updateWeightsOfCellForgetGate(int c) {

    for(int i = 0; i < I; ++i) {
        double gradient_i = 0; //the gradient of the inputs-forget_gate weights
        for(int t = 0; t < T; ++t) {
            gradient_i += delta_f(t, c)*x[t](i);
        }
        hiddenLayerNodes[c].w_ifg(i) -= ETA*gradient_i;
    }

    for(int i = 0; i < H; ++i) {
        double gradient_h = 0; //the gradient of the hidden_units-forget_gate weights
        for(int t = 0; t < T; ++t) {
            gradient_h += delta_f(t, c)*b_c[t](i);
        }
        hiddenLayerNodes[c].w_hfg(i) -= ETA*gradient_h;
    }

    for(int i = 0; i < H; ++i) {
        double gradient_c = 0; //the gradient of the cell_states - forget_gate weights
        for(int t = 0; t < T; ++t) {
            gradient_c += delta_f(t, c)*sc[t](i); /// assumption -- the input of this node is the output of interior cell s_c
        }
        hiddenLayerNodes[c].w_cfg(i) -= ETA*gradient_c;
    }

}



void BLSTM::updateWeightsOfCellOutputGate(int c) {

    for(int i = 0; i < I; ++i) {
        double gradient_i = 0; //the gradient of the inputs-output_gate weights
        for(int t = 0; t < T; ++t) {
            gradient_i += delta_o(t, c)*x[t](i);
        }
        hiddenLayerNodes[c].w_iog(i) -= ETA*gradient_i;
    }

    for(int i = 0; i < H; ++i) {
        double gradient_h = 0; //the gradient of the hidden_units-output_gate weights
        for(int t = 0; t < T; ++t) {
            gradient_h += delta_o(t, c)*b_c[t](i);
        }
        hiddenLayerNodes[c].w_hog(i) -= ETA*gradient_h;
    }

//    cout << "w_cog" << hiddenLayerNodes[c].w_cog(H-1) << "\n";
//    cout << "delta_o" << delta_o(0, 0) << "\n";
    for(int i = 0; i < H; ++i) {
        double gradient_c = 0; //the gradient of the cell_states - output_gate weights
        for(int t = 0; t < T; ++t) {
            gradient_c += delta_o(t, c)*sc[t](i); /// assumption -- the input of this node is the output of interior cell s_c
        }
        hiddenLayerNodes[c].w_cog(i) -= ETA*gradient_c;
//        cout << "w_cog " << hiddenLayerNodes[c].w_cog(i) << " ";
    }

//    cout << "w_cog-after" << hiddenLayerNodes[c].w_cog(H-1) << "\n";

}

void BLSTM::print() {
    cout << "for t = 10\n";
//    for(int t = 0; t < T; ++t)
    for(int i = 0; i < H; ++i)
        cout << b_c[10](i) << " ";
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
