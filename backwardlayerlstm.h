#ifndef BACKWARDLAYERLSTM_H
#define BACKWARDLAYERLSTM_H

#include "blstm.h"

class BackwardLayerLSTM : public BLSTM
{
    public:
//        BackwardLayerLSTM() : BLSTM() {}
        BackwardLayerLSTM(int hiddenUnitsNum): BLSTM(hiddenUnitsNum) {}
        virtual void forwardPass(vector< VectorXd > input);
        virtual void backwardPass(MatrixXd eps_c1);

};

#endif // BACKWARDLAYERLSTM_H
