/*
 * BLSTM.cpp
 *
 *  Created on: Mar 16, 2014
 *      Author: silvia
 */

#include "blstm.h"

BLSTM::BLSTM(int hiddenUnitsNum, vector< VectorXd > input) {
    int I = input[0].size();
    T = input.size();
    H = hiddenUnitsNum;
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
    MatrixXd delta_aux(T + 1, H);
    delta_o = delta_aux;
    delta_c = delta_aux;
    delta_f = delta_aux;
    delta_i = delta_aux;
    std::cout << "BLSTM - constructor\n";
    for(int i = 0; i < H; ++i)
    {
        LSTM unit(I, H);
        hiddenLayerNodes.push_back(unit);
    }
}

BLSTM::~BLSTM() {
    // TODO Auto-generated destructor stub
}


void BLSTM::forwardPass() {
    int T = x.size();
    VectorXd prev_b(H), prev_sc(H);
    std::cout << "forward - pass START\n";
    for(int t = 0; t < T; ++t) {
        VectorXd x_t = x[t];

        if(t == 0) {
            prev_b = VectorXd::Zero(H);
            prev_sc = VectorXd::Zero(H);
        } else {
            prev_b = b_c[t-1];
            prev_sc = sc[t-1];
        }
        for(int i = 0; i < H; ++i) {
            hiddenLayerNodes[i].startNewForwardPass(x_t, prev_b, prev_sc);
            LSTM memBlock = hiddenLayerNodes[i];
            a_ig[t](i) = memBlock.forwardPassInputGate();
            b_ig[t](i) = f(a_ig[t](i));
            a_fg[t](i) = memBlock.forwardPassForgetGate();
            b_fg[t](i) = f(a_fg[t](i));
            a_c[t](i) = memBlock.forwardPassCell();
            sc[t](i) = b_fg[t](i)*prev_sc[i] + b_ig[t](i)*g(a_c[t](i));
//            std::cout << "i=" << i << ":" << sc[t](i) << " ";
        }
        for(int i = 0; i < H; ++i) {
            LSTM memBlock = hiddenLayerNodes[i];
            a_og[t](i) = memBlock.forwardPassOutputGate(sc[t]);
            b_og[t](i) = f(a_og[t](i));
            std::cout << a_og[t](i) << " ";
            b_c[t](i) = b_og[t](i) * h(sc[t](i));
            std::cout << b_c[t](i) << " ";
        }
        std::cout << "\n";
    }

}

/**
 * - eps_c1: represents the component eps from the output layer (only for backward/forward -sense layer)
 */
void BLSTM::backwardPass(MatrixXd eps_c1) {
    int c, t, i;
//getEpsilonCTC() should be called in the neural_network class
    MatrixXd eps_c;
//    MatrixXd h_sc(T, H); //the function h applied to the cell-states
    //compute component of delta_o (the sum)
    VectorXd h_eps(T); // sum_c{1, H} [ h(sc)*eps_c ]
    MatrixXd eps_s(T, H); //  epsilon for cell states

    //initilize the delta variables at position T+1
    for(c = 0; c < H; ++c)
    {
        delta_i(T, c) = 0;
        delta_f(T, c) = 0;
        delta_o(T, c) = 0;
        delta_c(T, c) = 0;
    }

    for(t = T-1; t >= 0; t--) // iterate backward -- for the forward-hidden layer
    {
        h_eps(t) = 0;
        for(c = 0; c < H; ++c)
        {
            eps_c(t, c) = eps_c1(t, c);
            //sum over all recursively-hidden connections
            eps_c(t, c) += hiddenLayerNodes[c].w_hig[t]*delta_i(t+1, c);
            eps_c(t, c) += hiddenLayerNodes[c].w_hfg[t]*delta_f(t+1, c);
            eps_c(t, c) += hiddenLayerNodes[c].w_hog[t]*delta_o(t+1, c);
            eps_c(t, c) += hiddenLayerNodes[c].w_hc[t]*delta_c(t+1, c);
            h_eps(t) = h( sc[t](c) )*eps_c(t, c);
        }

        for(i = 0; i < H; ++i)
            delta_o(t, i) = f_derived(a_og[t](i)) * h_eps(t);

        for(i = 0; i < H; ++i) {
            //compute epsilon for states - sc
            eps_s(t, i) = b_og[t](i)*h_derived(sc[t](i))*eps_c(t, i) +
                    hiddenLayerNodes[c].w_cig(t)*delta_i(t+1, i) +
                    hiddenLayerNodes[c].w_cfg(t)*delta_f(t+1, i) +
                    hiddenLayerNodes[c].w_cog(t)*delta_o(t+1, i);
            if(t < T - 1) // to add nothing for T+1
                eps_s(t, i) += b_fg[t + 1](i)*eps_s(t + 1, i);

            //compute delta for the cells
            delta_c(t, i)  = b_ig[t](i)*g_derived(a_c[t][i])*eps_s(t, i);
        }

        //compute the sum-factor sum_c ( sc(t-1)*eps_s(t) )
        double sc_eps_s = 0;
        if(t > 0)
            sc_eps_s = sc[t-1].dot(eps_s.row(t));

        for(i = 0; i < H; ++i) {
            //delta for the forget gates
            delta_f(t, i) = f_derived(a_fg[t][i])*sc_eps_s;
        }

        double g_ac_eps = 0;
        for(c = 0; c < H; ++c)
            g_ac_eps += g(a_c[t](c))*eps_s(t, c);

        //delta for the input gates
        for(i = 0; i < H; ++i)
            delta_i(t, i) = f_derived( a_ig[t][i] ) * g_ac_eps;
    }

//    for(t = 0; t < T; ++t)
//        for(c = 0; c < H; ++c)
//            delta_o(t, c) = f_derived(a_og[t](c)) * h_eps(t);



}


vector<VectorXd> BLSTM::reserve(int s1, int s2) {
    vector<VectorXd> vec;
    for(int i = 0; i < s1; ++i) {
        VectorXd elem(s2);
        vec.push_back(elem);
    }
    return vec;
}
