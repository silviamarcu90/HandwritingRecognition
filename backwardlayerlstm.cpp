#include "backwardlayerlstm.h"

void BackwardLayerLSTM::forwardPass(vector< VectorXd > input) {
    BLSTM::initActivationsAndDelta(input);
    VectorXd prev_b(H), prev_sc(H); //prev is the next in the backwardLayer-traverse
    if(DEBUG)
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
            VectorXd prev_sc_c = VectorXd::Zero(C);
            prev_sc_c[0] = prev_sc[i];
            hiddenLayerNodes[i].startNewForwardPass(x_t, prev_b, prev_sc_c); //use the same weights            
//            if(t == 0)cout <<"DA" << hiddenLayerNodes[0].w_iig(0) << "\n";
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
    MatrixXd eps_s(T, H); //  epsilon for cell states

    if(DEBUG)
        std::cout << "backward - pass START ( <-- sense)\n";

    for(t = 0; t < T; t++) // iterate backward in the backward pass -- for the backward-hidden layer: 0->T
    {
        prev_t = t > 0 ? t-1 : 0; //for t=0, use delta(0, c) = 0 which is initialized at the beginning of the forward phase

        //compute component of delta_o (the sum)
        for(c = 0; c < H; ++c)
        {
            eps_c(t, c) = eps_c1(t, c);
            //sum over all recursively-hidden connections
            for(i = 0 ; i < H; ++i) { //4*H elements
                eps_c(t, c) += hiddenLayerNodes[c].w_hig[i]*delta_i(prev_t, i);
                eps_c(t, c) += hiddenLayerNodes[c].w_hfg[i]*delta_f(prev_t, i);
                eps_c(t, c) += hiddenLayerNodes[c].w_hog[i]*delta_o(prev_t, i);
                eps_c(t, c) += hiddenLayerNodes[c].w_hc[i]*delta_c(prev_t, i);
            }
        }

        //compute delta for the output gate
        for(i = 0; i < H; ++i) {
            delta_o(t, i) = f_derived(a_og[t](i)) * h( sc[t](i) ) * eps_c(t, i);
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

            /// NOT GOOD! we have considered links from all the states in the hidden layer to all => sum over all
            eps_s(t, i) += hiddenLayerNodes[i].w_cig(0)*delta_i(prev_t, i) +
                hiddenLayerNodes[i].w_cfg(0)*delta_f(prev_t, i) +
                hiddenLayerNodes[i].w_cog(0)*delta_o(t, i);
            if(t >= 1) // condition to add nothing for t=0
                eps_s(t, i) += b_fg[prev_t](i)*eps_s(prev_t, i);

            //compute delta for the cells
            delta_c(t, i)  = b_ig[t](i)*g_derived(a_c[t][i])*eps_s(t, i);
        }

        for(i = 0; i < H; ++i) {
            double sc_eps_s = 0;
            if(t < T-1) // condition to add zero for t = T-1
                sc_eps_s = sc[t+1](i)*eps_s(t, i); //{ sc(t-1)*eps_s(t) }

            //delta for the forget gates
            delta_f(t, i) = f_derived(a_fg[t][i])*sc_eps_s;
        }

        //compute the sum-factor sum_c { g(a_c(t)*eps_s(t) }


        for(i = 0; i < H; ++i) {
            //delta for the input gates
            delta_i(t, i) = f_derived( a_ig[t][i] ) * g(a_c[t](i))*eps_s(t, i);
//            cout << "[Back]delta_i(t, i): " << delta_i(t, i) <<  "\n";
        }
    }

}

void BackwardLayerLSTM::updateWeights(double eta) {
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


void BackwardLayerLSTM::updateWeightsOfCellState(int c) { //for a hidden memory cell

    for(int i = 0; i < I; ++i) { //for each input
        double gradient_i = 0; //the gradient of the inputs-input_gate weights
        for(int t = 0; t < T; ++t) {
            gradient_i += delta_c(t, c)*x[t](i);
        }
//        if(c == 0 && i == 0) cout << "GRADIENT " << gradient_i << "\n";
        updateOneWeight(ETA, hiddenLayerNodes[c].w_ic(i), hiddenLayerNodes[c].delta_w_ic(i), gradient_i);
//        hiddenLayerNodes[c].delta_w_ic(i) *= MIU;
//        hiddenLayerNodes[c].delta_w_ic(i) += -ETA*(1-MIU)*gradient_i;
//        hiddenLayerNodes[c].w_ic(i) += hiddenLayerNodes[c].delta_w_ic(i);
    }

    for(int i = 0; i < H; ++i) {
        double gradient_h = 0; //the gradient of the hidden_units-input_gate weights
        for(int t = 0; t < T-1; ++t) {
            gradient_h += delta_c(t, c)*b_c[t+1](i);
        }
//        if(c == 0 && i == 0) cout << "GRADIENT " << gradient_h << "\n";

        updateOneWeight(ETA, hiddenLayerNodes[c].w_hc(i), hiddenLayerNodes[c].delta_w_hc(i), gradient_h);

//        hiddenLayerNodes[c].w_hc(i) -= ETA*gradient_h;
    }

}

void BackwardLayerLSTM::updateWeightsOfCellInputGate(int c) {

    for(int i = 0; i < I; ++i) {
        double gradient_i = 0; //the gradient of the inputs-input_gate weights
        for(int t = 0; t < T; ++t) {
            gradient_i += delta_i(t, c)*x[t](i);
        }
//        if(c == 0 && i == 0) cout << "GRADIENT backpropagation " << gradient_i << "\n";

//        if(c == 2 && i == 1)
//            cout << "***gradient is: " << gradient_i << "\n";
        updateOneWeight(ETA, hiddenLayerNodes[c].w_iig(i), hiddenLayerNodes[c].delta_w_iig(i), gradient_i);

//        cout << "w_iig " << hiddenLayerNodes[c].w_iig(i) << " ";
    }
//    cout << "\n";
//    return;
    for(int i = 0; i < H; ++i) {
        double gradient_h = 0; //the gradient of the hidden_units-input_gate weights
        for(int t = 0; t < T-1; ++t) {
            gradient_h += delta_i(t, c)*b_c[t + 1](i);
        }
        updateOneWeight(ETA, hiddenLayerNodes[c].w_hig(i), hiddenLayerNodes[c].delta_w_hig(i), gradient_h);

//        cout << "w_hig " << hiddenLayerNodes[c].w_hig(i) << " ";
    }
//    cout << "\n";

//    for(int i = 0; i < H; ++i) {
        double gradient_c = 0; //the gradient of the cell_states - input_gate weights
        for(int t = 0; t < T-1; ++t) {
            gradient_c += delta_i(t, c)*sc[t + 1](c); /// assumption -- the input of this node is the output of interior cell s_c
        }
        updateOneWeight(ETA, hiddenLayerNodes[c].w_cig(0), hiddenLayerNodes[c].delta_w_cig(0), gradient_c);

//        cout << "w_cig " << hiddenLayerNodes[c].w_cig(i) << " "; //!? too big values (sometimes)
//    }
//    cout << "\n";

}


void BackwardLayerLSTM::updateWeightsOfCellForgetGate(int c) {

    for(int i = 0; i < I; ++i) {
        double gradient_i = 0; //the gradient of the inputs-forget_gate weights
        for(int t = 0; t < T; ++t) {
            gradient_i += delta_f(t, c)*x[t](i);
        }
        updateOneWeight(ETA, hiddenLayerNodes[c].w_ifg(i), hiddenLayerNodes[c].delta_w_ifg(i), gradient_i);
    }

    for(int i = 0; i < H; ++i) {
        double gradient_h = 0; //the gradient of the hidden_units-forget_gate weights
        for(int t = 0; t < T-1; ++t) {
            gradient_h += delta_f(t, c)*b_c[t + 1](i);
        }
        updateOneWeight(ETA, hiddenLayerNodes[c].w_hfg(i), hiddenLayerNodes[c].delta_w_hfg(i), gradient_h);
    }

//    for(int i = 0; i < H; ++i) {
        double gradient_c = 0; //the gradient of the cell_states - forget_gate weights
        for(int t = 0; t < T-1; ++t) {
            gradient_c += delta_f(t, c)*sc[t + 1](c); /// assumption -- the input of this node is the output of interior cell s_c
        }
//        if(c == 0) cout << "GRADIENT " << gradient_c << "\n";

        updateOneWeight(ETA, hiddenLayerNodes[c].w_cfg(0), hiddenLayerNodes[c].delta_w_cfg(0), gradient_c);
//    }

}



void BackwardLayerLSTM::updateWeightsOfCellOutputGate(int c) {

    for(int i = 0; i < I; ++i) {
        double gradient_i = 0; //the gradient of the inputs-output_gate weights
        for(int t = 0; t < T; ++t) {
            gradient_i += delta_o(t, c)*x[t](i);
        }
//        if(c == 0 && i == 0)
//            cout << "GRADIENT " << gradient_i << "\n";
        updateOneWeight(ETA, hiddenLayerNodes[c].w_iog(i), hiddenLayerNodes[c].delta_w_iog(i), gradient_i);
    }
//    return;

    for(int i = 0; i < H; ++i) {
        double gradient_h = 0; //the gradient of the hidden_units-output_gate weights
        for(int t = 0; t < T-1; ++t) {
            gradient_h += delta_o(t, c)*b_c[t + 1](i);
        }
//        if(c == 0 && i == 0)
//            cout << "GRADIENT " << gradient_h << "\n";
        updateOneWeight(ETA, hiddenLayerNodes[c].w_hog(i), hiddenLayerNodes[c].delta_w_hog(i), gradient_h);

    }
//    return;
//    cout << "w_cog" << hiddenLayerNodes[c].w_cog(H-1) << "\n";
//    cout << "delta_o" << delta_o(0, 0) << "\n";
//    for(int i = 0; i < H; ++i) {
        double gradient_c = 0; //the gradient of the cell_states - output_gate weights
        for(int t = 0; t < T; ++t) { //state at time t is correct for both layers
            gradient_c += delta_o(t, c)*sc[t](c); /// assumption -- the input of this node is the output of interior cell s_c
        }
//        if(c == 0) cout << "GRADIENT " << gradient_c << "\n";

        updateOneWeight(ETA, hiddenLayerNodes[c].w_cog(0), hiddenLayerNodes[c].delta_w_cog(0), gradient_c);
//        cout << "w_cog " << hiddenLayerNodes[c].w_cog(i) << " ";
//    }

//    cout << "w_cog-after" << hiddenLayerNodes[c].w_cog(H-1) << "\n";

}

