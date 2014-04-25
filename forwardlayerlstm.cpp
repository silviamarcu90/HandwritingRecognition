#include "forwardlayerlstm.h"

void ForwardLayerLSTM::forwardPass(vector< VectorXd > input) {
    BLSTM::initActivationsAndDelta(input);
//    cout << "T = " << T << "\n\n";
    VectorXd prev_b(H), prev_sc(H);
    if(DEBUG) std::cout << "forward - pass START\n";
    for(int t = 0; t < T; ++t) {
        VectorXd x_t = x[t]; //for each input column (from the T length sequence)

        if(t == 0) {
            prev_b = VectorXd::Zero(H);
            prev_sc = VectorXd::Zero(H);
        } else {
            prev_b = b_c[t-1];
            prev_sc = sc[t-1];
        }
        for(int i = 0; i < H; ++i) { //for each neuron in the hidden layer
            VectorXd prev_sc_c = VectorXd::Zero(C);
            prev_sc_c[0] = prev_sc[i];
            hiddenLayerNodes[i].startNewForwardPass(x_t, prev_b, prev_sc_c); //use the same weights
            LSTM memBlock = hiddenLayerNodes[i];
            a_ig[t](i) = memBlock.forwardPassInputGate();
            b_ig[t](i) = f(a_ig[t](i));
            a_fg[t](i) = memBlock.forwardPassForgetGate();
            b_fg[t](i) = f(a_fg[t](i));
            a_c[t](i) = memBlock.forwardPassCell();
            sc[t](i) = b_fg[t](i)*prev_sc[i] + b_ig[t](i)*g(a_c[t](i));
//            std::cout << "i=" << i << ":" << sc[t](i) << " ";
            VectorXd sc_c = VectorXd::Zero(C);
            sc_c[0] = sc[t](i);
            a_og[t](i) = memBlock.forwardPassOutputGate(sc_c);
            b_og[t](i) = f(a_og[t](i));
//            std::cout << "a_og" << a_og[t](i) << " ";
            b_c[t](i) = b_og[t](i) * h(sc[t](i)); //cell outputs
//            std::cout << "bc_forward" << b_c[t](i) << " ";
        }
//        std::cout << "\n";
    }

}

/**
 * - eps_c1: represents the component eps from the output layer (only for backward/forward -sense layer)
 */
void ForwardLayerLSTM::backwardPass(MatrixXd eps_c1) {
    int c, t, i;
    MatrixXd eps_c(T, H);
    //compute component of delta_o (the sum)
    MatrixXd eps_s(T, H); //  epsilon for cell states
    if(DEBUG) std::cout << "backward - pass START (-> sense) [" << T << "]\n";

    for(t = T-1; t >= 0; t--) // iterate backward -- for the forward-hidden layer
    {
        for(c = 0; c < H; ++c)
        {
            eps_c(t, c) = eps_c1(t, c);
            //sum over all recursively-hidden connections
            for(i = 0 ; i < H; ++i) { //4*H elements (i is g from the formulas)
                eps_c(t, c) += hiddenLayerNodes[c].w_hig[i]*delta_i(t+1, i);
                eps_c(t, c) += hiddenLayerNodes[c].w_hfg[i]*delta_f(t+1, i);
                eps_c(t, c) += hiddenLayerNodes[c].w_hog[i]*delta_o(t+1, i);
                eps_c(t, c) += hiddenLayerNodes[c].w_hc[i]*delta_c(t+1, i);
            }
//            sum_h_eps[t] += h( sc[t](c) )*eps_c(t, c);
//            cout << "h_eps: " << sum_h_eps(t) << " eps_c: " << eps_c(t, c) <<  "\n";

        }

        //compute delta for the output gate
        for(i = 0; i < H; ++i) {
            delta_o(t, i) = f_derived(a_og[t](i)) * h( sc[t](i) ) * eps_c(t, i) ;
//            cout << "delta_o(t, i) " << delta_o(t, i) << "\n"; ///???is nan why??
        }
//        if(t == 1) {
//            cout << "f_derived(a_og)" << f_derived(a_og[t](H-1)) << "\n";
//            cout << "h_eps" << h_eps(t) << " eps_c: " << eps_c(t, H-1) <<  "\n";
//        }


        for(i = 0; i < H; ++i) {
            //compute epsilon for each state - sc
            eps_s(t, i) = b_og[t](i)*h_derived(sc[t](i))*eps_c(t, i);

            /// NOT GOOD!! we have considered links from all the states in the hidden layer to all => sum over all
//           for(int j = 0; j < H; ++j)
            eps_s(t, i) += hiddenLayerNodes[i].w_cig(0)*delta_i(t+1, i) +
                hiddenLayerNodes[i].w_cfg(0)*delta_f(t+1, i) +
                hiddenLayerNodes[i].w_cog(0)*delta_o(t, i);
            if(t < T - 1) // condition to add nothing for T+1
                eps_s(t, i) += b_fg[t + 1](i)*eps_s(t + 1, i);
//            cout << "eps_s(t, i): " << eps_s(t, i) <<  "\n";

            //compute delta for the cells
            delta_c(t, i)  = b_ig[t](i)*g_derived(a_c[t][i])*eps_s(t, i);
//            cout << "delta_c(t, i): " << delta_c(t, i) <<  "\n";
        }

        //compute the sum-factor sum_c { sc(t-1)*eps_s(t) }
//        double sc_eps_s = 0;
//        if(t > 0)
//            sc_eps_s = sc[t-1](i).dot(eps_s.row(t));

        for(i = 0; i < H; ++i) {
            double sc_eps_s = 0; //compute { sc(t-1)*eps_s(t) }
            if(t > 0)
                sc_eps_s = sc[t-1](i)*eps_s(t, i);
            //delta for the forget gates
            delta_f(t, i) = f_derived(a_fg[t][i])*sc_eps_s;
//            cout << "delta_f(t, i): " << delta_f(t, i) <<  "\n";
        }

         //        cout << "sum g(a_c)*eps_s(t,c) " << g_ac_eps << "\n";
        //delta for the input gates
        for(i = 0; i < H; ++i) {
            double g_ac_eps = 0; //{ g(a_c(t)*eps_s(t) }
            g_ac_eps = g(a_c[t](i))*eps_s(t, i);
            delta_i(t, i) = f_derived( a_ig[t][i] ) * g_ac_eps; //--sometimes too big values => normalize inputs in (0,1)?
//            cout << "delta_i(t, i): " << delta_i(t, i) <<  "\n";
        }
    }

}
