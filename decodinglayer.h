#ifndef DECODINGLAYER_H
#define DECODINGLAYER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include "../Eigen/Core"
#include "log.h"

using namespace Eigen;
using namespace std;

/**
 *  object to define the score for a part of a word from the dictionary: token
 */
struct token {
    vector<string> history;
    double score;
};

/**
 *  object to define the full score of a word from the dictionary
 */
struct word {
    string label; //the real word from the dictionary (model language) used
    vector< vector <token> > tok; //matrix of tokens
};

class DecodingLayer
{
    int K, T;
    vector<string> dictionary;
    vector< word > words; // for each word in the dictionary store a matrix of scores: TxK
    MatrixXd y; // the softmax output from the output layer: T x K
    map<char, int> alphabet;
    void initDictionary(string dictionaryPath);
    word initTokensWord(string dictWord);
    string createExtendedLabel(string l);
public:

    DecodingLayer(MatrixXd y1, string dictionaryPath, map<char, int> alphabet1) :
        y(y1), alphabet(alphabet1)
    {
        initDictionary(dictionaryPath);
        K = y.cols();
        T = y.rows();
    }

    void init();
    vector<string> getDecodedLabels();
    token getHighestScoreOutputToken(int t);

};

#endif // DECODINGLAYER_H
