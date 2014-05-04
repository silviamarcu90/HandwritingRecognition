#ifndef FORWARDLAYERLSTM_H
#define FORWARDLAYERLSTM_H

#include "blstm.h"

class ForwardLayerLSTM : public BLSTM
{
    public:
//        ForwardLayerLSTM() : BLSTM() {}
        ForwardLayerLSTM(int hiddenUnitsNum): BLSTM(hiddenUnitsNum) {}
        ForwardLayerLSTM(int hiddenUnitsNum, istream& fin): BLSTM(hiddenUnitsNum, fin) {}
        virtual void forwardPass(vector< VectorXd > input);
        virtual void backwardPass(MatrixXd eps_c1);
        virtual void updateWeightsOfCellInputGate(int c);
        virtual void updateWeightsOfCellForgetGate(int c);
        virtual void updateWeightsOfCellOutputGate(int c);
        virtual void updateWeightsOfCellState(int c);
        virtual void updateWeights(double ETA);

};

#endif // FORWARDLAYERLSTM_H
