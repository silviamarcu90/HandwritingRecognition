#ifndef BACKWARDLAYERLSTM_H
#define BACKWARDLAYERLSTM_H

#include "blstm.h"

class BackwardLayerLSTM : public BLSTM
{
    public:
//        BackwardLayerLSTM() : BLSTM() {}
        BackwardLayerLSTM(int hiddenUnitsNum): BLSTM(hiddenUnitsNum) {}
        BackwardLayerLSTM(int hiddenUnitsNum, istream& fin): BLSTM(hiddenUnitsNum, fin) {}
        virtual void forwardPass(vector< VectorXd > input);
        virtual void backwardPass(MatrixXd eps_c1);
        virtual void updateWeightsOfCellInputGate(int c);
        virtual void updateWeightsOfCellForgetGate(int c);
        virtual void updateWeightsOfCellOutputGate(int c);
        virtual void updateWeightsOfCellState(int c);
        virtual void updateWeights(double ETA);

};

#endif // BACKWARDLAYERLSTM_H
