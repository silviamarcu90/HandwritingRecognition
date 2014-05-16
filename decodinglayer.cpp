#include "decodinglayer.h"

/**
 * Function used in constructor to init the dictionary D of words considered
 */
void DecodingLayer::initDictionary(string dictionaryPath) {

    string w;
    //read words from file
    ifstream in(dictionaryPath);
    if(in == NULL) {
        cout << "Error when opening file!\n";
        return;
    }

    while(in >> w)
        dictionary.push_back(w);
}

/**
 * Initialization phase of DECODING
 */
void DecodingLayer::init() {
    int dictSize;
    dictSize = dictionary.size();
    for(int i = 0; i < dictSize; ++i) {
        string wordFromDict = dictionary[i];
        word w = initTokensWord(wordFromDict);
        words.push_back(w);
//        if(i == 10) {
//            int S = wordFromDict.size()*2 + 1;
//            for(int i = 0; i < S+2; ++i) //S+2 because I need the special indices: -1 (output token) and 0 (input token)
//            {
//                for(int j = 0; j < T; ++j)
//                {
//                    token tok = w.tok[i][j];
//                    cout << tok.score << " ";
//                }
//                cout << "\n";
//            }
//        }
    }
}

word DecodingLayer::initTokensWord(string dictWord) {
    word ret;
    ret.label = dictWord;
    int S = 2*dictWord.size() + 1;

    for(int i = 0; i < S+2; ++i) //S+2 because I need the special indices: -1 (output token) and 0 (input token)
    {
        vector<token> vecTok; //a row of the matrix -- for a specific segment _i_
        for(int j = 0; j < T; ++j)
        {
            token tok;
            tok.score = logZero; //init all with ln 0 = -inf
            vecTok.push_back(tok); //(-inf, empty-set)
        }
        ret.tok.push_back(vecTok); //add the row to the matrix
    }
    //tok(w, s=1, t=1)
    ret.tok[2][0].score = safe_log( y(0, alphabet[' ']) );
    ret.tok[2][0].history.push_back(dictWord);

    //tok(w, 2, 1)
    ret.tok[3][0].score = safe_log( y(0, alphabet[dictWord[0]]) );
    ret.tok[3][0].history.push_back(dictWord);

    if(dictWord.size() == 1) //init tok(w, -1, 1)
        ret.tok[1][0] = ret.tok[3][0];
    else {
        token tok;
        tok.score = logZero;
        ret.tok[1][0] = tok;
    }
    return ret;
}

vector<string> DecodingLayer::getDecodedLabels() {
    int nbWords = words.size();

    for(int t = 1; t < T; ++t)
    {
        token highestOutputToken = getHighestScoreOutputToken(t);

        for(int i = 0; i < nbWords; ++i)
        {
            words[i].tok[0][t] = highestOutputToken;
            words[i].tok[0][t].history.push_back(words[i].label); //add w to tok(w, 0, t) history

            string w_prime = createExtendedLabel(words[i].label);
            int S = w_prime.size();
            for(int s = 0; s < S; ++s)
            {
//                vector<token> P; //don't used -- compute maxTok directly
                token maxTok = words[i].tok[s+2][t-1];
//                P.push_back(words[i].tok[s+2][t-1]);
//                P.push_back(words[i].tok[s+1][t-1]);
                int prevSeg = //(s == 0) ? 0 :
                              s+1; // s+1 is different for the first segment !! better results without condition

                if(words[i].tok[prevSeg][t-1].score > maxTok.score)
                    maxTok = words[i].tok[prevSeg][t-1];

                if(w_prime[s] != ' ' && s >= 2 && w_prime[s-2] != w_prime[s])
                {
//                    P.push_back(words[i].tok[s][t-1]);
                    if(words[i].tok[s][t-1].score > maxTok.score)
                        maxTok = words[i].tok[s][t-1];
                }
                words[i].tok[s+2][t] = maxTok; //highest scoring token from set P
                words[i].tok[s+2][t].score += safe_log( y(t, alphabet[w_prime[s]]) );
            }

            //compute the highest score
            token maxTok = words[i].tok[S + 1][t];
            if(words[i].tok[S][t].score > maxTok.score)
                maxTok = words[i].tok[S][t];
            words[i].tok[1][t] = maxTok;
        }
    }


    //output the top 10 bestwords
    sortVector(words);
    token maxTok = words[0].tok[1][T-1];
    string bestword = words[0].label;

    cout << "+++ ";
    for(int i = 0; i < maxTok.history.size(); ++i)
        cout << maxTok.history[i] <<  " ";
    cout << "++++\n";

    vector<string> result;
    for(int i = 0; i < 10; ++i)
        result.push_back(words[i].label);

    return result;//maxTok.history;
}

token DecodingLayer::getHighestScoreOutputToken(int t) {
    token result = words[0].tok[1][t-1];
    int nbWords = words.size();

    for(int i = 1; i < nbWords; ++i) {
        if(words[i].tok[1][t-1].score > result.score)
        {
            result = words[i].tok[1][t-1];
        }
    }
//    cout << "res - score: " << result.score << "\n";
    return result;
}

string DecodingLayer::createExtendedLabel(string l) {
    string l_prime = " ";
    for(int i = 0; i < l.size(); ++i)
    {
        l_prime += string(1, l[i]);
        l_prime += string(1, ' ');
    }
//    cout << "l_prime = " << l_prime <<"\n";
    return l_prime;
}


bool compareByScore(const word &a, const word &b)
{
    int T = a.tok[1].size();
    return a.tok[1][T-1].score > b.tok[1][T-1].score;
}

void DecodingLayer::sortVector(vector<word> &wordsVec)
{
    sort(wordsVec.begin(), wordsVec.end(), compareByScore);
}



//int levenstein(char *a, char *b){
//    int cost[32][32] = {};
//    int min = 10000;
//    int i,j;
//    m = strlen(a) + 1;
//    n = strlen(b) + 1;

//    printf("m = %d; n = %d\n", m, n);
////    cost = (int**)malloc(m);
////    for(i=0; i< m; i++)
////        cost[i] = (int*)malloc(n);
//    //initialization
//    for(i=0; i < m; i++) // completez prima coloana
//        cost[i][0] = i;
//    for(i=0; i < n; i++) // completez prima linie
//        cost[0][i] = i;
//    for(i=1; i < m; i++)
//        for(j=1; j < n; j++)
//            if(a[i-1]==b[j-1])
//                cost[i][j] = cost[i-1][j-1];
//            else if(i >= 2 && j >= 2
//                    && a[i-2] == b[j-1]
//                    && a[i-1] == b[j-2]){ //compute min
//                min=cost[i-2][j-2];
//                if(min>cost[i-1][j])
//                    min=cost[i-1][j];
//                if(min>cost[i][j-1])
//                    min=cost[i][j-1];
//                cost[i][j] = 1 + min;
//            }
//            else{ //compute min
//                min=cost[i-1][j-1];
//                if(min>cost[i-1][j])
//                    min=cost[i-1][j];
//                if(min>cost[i][j-1])
//                    min=cost[i][j-1];
//                cost[i][j] = 1 + min;
//            }
//    return cost[m-1][n-1];
//}
