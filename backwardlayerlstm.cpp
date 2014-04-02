#include "backwardlayerlstm.h"

void BackwardLayerLSTM::forwardPass(vector< VectorXd > input) {
    BLSTM::initActivationsAndDelta(input);
    VectorXd prev_b(H), prev_sc(H); //prev is the next in the backwardLayer-traverse
    std::cout << "forward - pass START in backward layer\n";

    /// the order of traversing the sequence makes the difference of the forward-layer
    for(int t = T-1; t >= 0; t--) {
        VectorXd x_t = x[t]; //for each input column (from the T length sequence)

        if(t == T - 1) {
            prev_b = VectorXd::Zero(H);
            prev_sc = VectorXd::Zero(H);
        } else {
            prev_b = b_c[t + 1];
            prev_sc = sc[t + 1];
        }
        for(int i = 0; i < H; ++i) {
            hiddenLayerNodes[i].startNewForwardPass(x_t, prev_b, prev_sc); //use the same weights
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
//            std::cout << a_og[t](i) << " ";
            b_c[t](i) = b_og[t](i) * h(sc[t](i));
//            std::cout << "bc_backward" << b_c[t](i) << " ";
        }
//        std::cout << "\n";
    }

}

/**
 * - eps_c1: represents the component eps from the output layer (only for backward/forward -sense layer)
 */
void BackwardLayerLSTM::backwardPass(MatrixXd eps_c1) {
    int c, t, i, prev_t;
    MatrixXd eps_c(T, H);
    VectorXd h_eps(T); // sum_c{1, H} [ h(sc)*eps_c ]
    MatrixXd eps_s(T, H); //  epsilon for cell states

    std::cout << "backward - pass START ( <-- sense)\n";

    //initilize the delta variables at position T+1 --- they are already initialized
//    for(c = 0; c < H; ++c)
//    {
//        delta_i(T, c) = 0;
//        delta_f(T, c) = 0;
//        delta_o(T, c) = 0;
//        delta_c(T, c) = 0;
//    }

    for(t = 0; t < T; t++) // iterate backward in the backward pass -- for the backward-hidden layer: 0->T
    {
        h_eps[t] = 0;
        prev_t = t > 0 ? t-1 : 0; //for t=0, use delta(0, c) = 0 which is initialized at the beginning of the forward phase

        //compute component of delta_o (the sum)
        for(c = 0; c < H; ++c)
        {
            eps_c(t, c) = eps_c1(t, c);
            //sum over all recursively-hidden connections
            for(i = 0 ; i < H; ++i) { //4*H elements
                eps_c(t, c) += hiddenLayerNodes[c].w_hig[i]*delta_i(prev_t, c);
                eps_c(t, c) += hiddenLayerNodes[c].w_hfg[i]*delta_f(prev_t, c);
                eps_c(t, c) += hiddenLayerNodes[c].w_hog[i]*delta_o(prev_t, c);
                eps_c(t, c) += hiddenLayerNodes[c].w_hc[i]*delta_c(prev_t, c);
            }
            h_eps[t] += h( sc[t](c) )*eps_c(t, c);
        }

        //compute delta for the output gate
        for(i = 0; i < H; ++i) {
            delta_o(t, i) = f_derived(a_og[t](i)) * h_eps(t);
//            if(t == 1)
//                cout << "delta_o(1,:) " << delta_o(t, i) << "\n";
        }

//        if(t == 1) {
//            cout << "f_derived(a_og)" << f_derived(a_og[t](H-1)) << "\n";
//            cout << "h_eps" << h_eps(t) << " eps_c: " << eps_c(t, H-1) <<  "\n";
//        }
        for(i = 0; i < H; ++i) {
            //compute epsilon for each state - sc
            eps_s(t, i) = b_og[t](i)*h_derived(sc[t](i))*eps_c(t, i);

            /// we have considered links from all the states in the hidden layer to all => sum over all
            for(int j = 0; j < H; ++j)
                eps_s(t, i) += hiddenLayerNodes[i].w_cig(j)*delta_i(prev_t, i) +
                    hiddenLayerNodes[i].w_cfg(j)*delta_f(prev_t, i) +
                    hiddenLayerNodes[i].w_cog(j)*delta_o(prev_t, i);
            if(t >= 1) // condition to add nothing for t=0
                eps_s(t, i) += b_fg[prev_t](i)*eps_s(prev_t, i);

            //compute delta for the cells
            delta_c(t, i)  = b_ig[t](i)*g_derived(a_c[t][i])*eps_s(t, i);
        }

        //compute the sum-factor sum_c { sc(t-1)*eps_s(t) }
        double sc_eps_s = 0;
        if(t < T-1) // condition to add zero for t = T-2
            sc_eps_s = sc[t+1].dot(eps_s.row(t));

        for(i = 0; i < H; ++i) {
            //delta for the forget gates
            delta_f(t, i) = f_derived(a_fg[t][i])*sc_eps_s;
        }

        //compute the sum-factor sum_c { g(a_c(t)*eps_s(t) }
        double g_ac_eps = 0;
        for(c = 0; c < H; ++c)
            g_ac_eps += g(a_c[t](c))*eps_s(t, c);

        //delta for the input gates
        for(i = 0; i < H; ++i) {
            delta_i(t, i) = f_derived( a_ig[t][i] ) * g_ac_eps;
//            cout << "[Back]delta_i(t, i): " << delta_i(t, i) <<  "\n";
        }
    }

}
