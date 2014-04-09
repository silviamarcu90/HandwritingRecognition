#ifndef INPUTDATA_H
#define INPUTDATA_H

#include "featureextractor.h"

class InputData
{
public:
    string imagePath;
    vector<VectorXd> featuresSeq;
    string target;
    InputData();
};

#endif // INPUTDATA_H
