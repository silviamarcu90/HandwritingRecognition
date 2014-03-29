#ifndef FORWARDLAYERLSTM_H
#define FORWARDLAYERLSTM_H

#include "blstm.h"

class ForwardLayerLSTM : public BLSTM
{
    public:
//        ForwardLayerLSTM() : BLSTM() {}
        ForwardLayerLSTM(int hiddenUnitsNum): BLSTM(hiddenUnitsNum) {}
        virtual void forwardPass(vector< VectorXd > input);
        virtual void backwardPass(MatrixXd eps_c1);
};

#endif // FORWARDLAYERLSTM_H
