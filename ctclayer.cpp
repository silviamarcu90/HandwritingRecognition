#include "ctclayer.h"

CTCLayer::CTCLayer(int K, int H) {
    this->H = H;
    this->K = K;

    initWeights(); /// init w variable (vector with 2 matrices: one for the forward-hidden layer, one for the backward-hidden)

    initAlphabet();
}

CTCLayer::CTCLayer() {

}


CTCLayer::~CTCLayer() {

}

void CTCLayer::initActivations() {

    a =  MatrixXd::Zero(T, K);

    y = MatrixXd::Zero(T, K);

    delta_k = MatrixXd::Zero(T, K);

}

/**
 * function used to compute the outputs of the CTC layer, i.e. y_k
 * given the outputs received from the 2 hidden layers (forward_b and backward_b)
 */
void CTCLayer::forwardPass(int T, string label, vector<VectorXd> forward_b, vector<VectorXd> backward_b) {

    VectorXd eSum(T);
    MatrixXd exp_a;


    this->T = T; /// length of the input sequence
    this->l = label; /// output label

    /// initialize inputs with the outputs of the hidden layers
    this->forward_b = forward_b;
    this->backward_b = backward_b;

    initActivations(); //new activations for each input label
    createExtendedLabel(); //generate extended label l_prime with spaces between every 2 characters

    //w[i] -- matrix K x H
    //forward_b[t] -- is a vector with the outputs of the hidden layer - H - at time _t_
    for(int t = 0; t < T; ++t) {
        a.row(t) = w[0]*forward_b[t] + w[1]*backward_b[t]; //a[t] is a vector of K
//        for(int k = 0; k < K; ++k)
//            cout << a(t, k) << " ";
//        cout << "\n";
    }

    // apply softmax function to compute y_k at each timestep t
    exp_a = compute_exp(a);
    eSum = exp_a.rowwise().sum(); //compute the sum of the elements of each row
    for(int t = 0; t < T; ++t) {
        y.row(t) = exp_a.row(t)/eSum[t];
//        for(int k = 0; k < K; ++k)
//            cout << y(t, k) << " ";
//        cout << "\n";
    }
}

/**
 * Run the backward pass and compute the residuals
 */
void CTCLayer::backwardPass() {

    computeForwardVariable();
    computeBackwardVariable();

    double logProbab = computeObjectiveFunction();

    //print the objective function: -ln(z|x)
    cout << " logProbab is " << -logProbab << "\n";
//    for(int t = 0; t < T; ++t)
//    {
//        double p = 0;
//        for(int u = 0; u < Uprime; ++u)
//            p += alpha(t, u) * beta(t, u);
//        p = (p == 0) ? 1e-10 : p; /// to avoid 0 probabilities!!!!! SOLUTION?
//        cond_probabs.push_back(p); // p(z|x) at time t
////        cout << "p = " << p << "\n";
//    }


    //compute residuals: delta_k(t, k)
    for(int t = 0; t < T; ++t) {
        for(int k = 0; k < K; ++k)
        {
            double sumProbab = computeProbability(k, t);//get the sum of probabilities of the k letter from the alphabet

            delta_k(t, k) = y(t, k) - safe_exp(log_divide(sumProbab, logProbab));
//            cout << delta_k(t, k) << " ";
        }
//        cout << "\n";
    }

    // update objective function - ctc-error
    trainError += -logProbab;
    cout << " logProbab" << logProbab << "\n";// << "; exp(logProbab) " << safe_exp(logProbab) << "\n";

}

/**
 * Compute the objective function L = - ln p(z|x) for the current input
 */
double CTCLayer::computeObjectiveFunction() {
    int Uprime = l_prime.size();

    //compute p(z|x) for t = T-1 using only alpha
    int t = T - 1;
    double logProbab = log_add( alpha(t, Uprime-1), alpha(t, Uprime-2) ); //page 57, formula 7.4
//    for(int u = 1; u < Uprime; ++u)
//        logProbab = log_add( logProbab, alpha(t, u) );

    return logProbab;
}

/**
 * compute error (in order to be used for validation)
 */
double CTCLayer::computeError() {
    computeForwardVariable();
    computeBackwardVariable();

    return - computeObjectiveFunction(); //-ln [p(z|x)]
}


/**
 * Compute and return the eps_cell term used for the residuals in the hidden layer
 * *function used in the backward pass by the BLSTM cells
 */
vector<MatrixXd> CTCLayer::getEpsilonCTC() {
    MatrixXd eps_f(T, H), eps_b(T, H);
    eps_c1.resize(0); //reset vector!
    for(int t = 0; t < T; ++t)
    {
        eps_f.row(t) = delta_k.row(t) * w[0];
        eps_b.row(t) = delta_k.row(t) * w[1];
//        if(t == 0) {
//            cout << "eps_f " << eps_f(t, H-1) << " eps_b: " << eps_b(t, H-1) <<  "\n";
//        }
    }
    eps_c1.push_back(eps_f); //for the forward layer
    eps_c1.push_back(eps_b); // for the backward layer

    return eps_c1;
}

void CTCLayer::updateWeights(double ETA) {
    MatrixXd delta_w_forward = MatrixXd::Zero(K, H);
    MatrixXd delta_w_backward = MatrixXd::Zero(K, H);

    for(int t = 0; t < T; ++t) {
        for(int k = 0; k < K; ++k) {
            for(int h = 0; h < H; ++h)
            {
                delta_w_forward.coeffRef(k, h) += delta_k(t, k)*forward_b[t](h);
                delta_w_backward.coeffRef(k, h) += delta_k(t, k)*backward_b[t](h);
            }
        }
    }

    //DEBUG gradient
//    cout << "gradient is : " << delta_w_forward(1, 3) << "\n";

    for(int k = 0; k < K; ++k) {
        for(int h = 0; h < H; ++h) {
//            cout << delta_w_forward(k, h) << " "; //sometimes I get too big values!!: eg. 63
//            w[0].coeffRef(k, h) -= ETA*delta_w_forward(k, h);
//            w[1].coeffRef(k, h) -= ETA*delta_w_backward(k, h);
//            cout << "w[0] " << w[0](k, h) << " ";
//            cout << "w[1] " << w[1](k, h) << " ";
        }
//        cout << "\n";
    }

}

/**
 * compute the probability of all the occurences of label k in the extended target
 * at time t in the input sequence
 */
double CTCLayer::computeProbability(int k, int t) {
    double probab = safe_log(0);
    char c = getKeyByValue(k);
    int Uprime = l_prime.size();

    for(int u = 0; u < Uprime; ++u)
        if(l_prime[u] == c)
            probab = log_add( probab, log_multiply( alpha(t, u), beta(t, u) ));
    return probab;
}

/**
 * get the key in alphabet-map knowing the value (k)
 */
char CTCLayer::getKeyByValue(int k) {
    map<char,int>::iterator it;
    for(it = alphabet.begin(); it != alphabet.end(); ++it)
        if(it->second == k)
            return it->first;
    return 0;
}

/**
 * Compute the forward variable: alpha [TxUprime]
 */
void CTCLayer::computeForwardVariable() {
    int t, u;
    int Uprime = l_prime.size();
    alpha = MatrixXd::Zero(T, Uprime);;

    //initialization
    alpha(0, 0) = safe_log( y(0, alphabet[' ']) );
    alpha(0, 1) = safe_log( y(0, alphabet[l[0]]) );
    for(u = 2; u < Uprime; ++u)
        alpha(0, u) = safe_log(0);

    //update for each timestep
    for(t = 1; t < T; ++t) {
        for(u = 0; u < Uprime; ++u)
        {
//            if(u < Uprime - 2*(T-t) - 1)
//                alpha(t, u) = 0;
//            else {
                int begin_idx = f_u(u);
                alpha(t, u) = alpha(t-1, begin_idx);
                for(int i = begin_idx + 1; i <= u; ++i)
                    alpha(t, u) = log_add( alpha(t, u), alpha(t-1, i) );
                if(t == 1 && u == 1)
                    cout << "for letter: " << alphabet[l_prime[u]] << ": " << y(t, alphabet[l_prime[u]]);
                alpha(t, u) += safe_log( y(t, alphabet[l_prime[u]]) );
//            }
//            cout << alpha(t, u) << " ";
        }
//        cout << "\n";
    }
}


/**
 * Compute the backward variable: beta(t, u)
 */
void CTCLayer::computeBackwardVariable() {
    int t, u;
    int Uprime = l_prime.size();
    MatrixXd beta_aux(T, Uprime);
    beta = beta_aux;

    //initialization
    beta(T-1, Uprime-1) = beta(T-1, Uprime - 2) = safe_log(1);
    for(u = 0; u < Uprime - 2; ++u)
        beta(T-1, u) = safe_log(0);

    for(t = T-2; t >=0; t--) {
        for(u = 0; u < Uprime; ++u)
        {
//            if(u > 2*t)
//                beta(t, u) = 0;
//            else {
                int upperBound = g_u(u);
                beta(t,u) = beta(t+1, u) + safe_log ( y(t+1, alphabet[l_prime[u]]) );
                for(int i = u + 1; i <= upperBound; ++i)
                    beta(t,u) = log_add( beta(t, u), beta(t+1, i) + safe_log( y(t+1, alphabet[l_prime[i]])) );
//            }
//            cout << beta(t, u) << " ";
        }
//        cout << "\n";
    }
}

/**
 * internal function - helper for computing the forward variable
 */
double CTCLayer::f_u(int u) {
    if( (u > 0 && l_prime[u] == ' ')
            || (u >= 2 && l_prime[u-2] == l_prime[u]))
        return u-1;
    return (u >= 2) ? (u - 2) : 0;
}

/**
 * internal function - helper for computing the backward variable
 */
double CTCLayer::g_u(int u) {
    int Uprime = l_prime.size();
    if( (u+1 < Uprime && l_prime[u] == ' ')
            || (u+2 < Uprime && l_prime[u+2] == l_prime[u]))
        return u + 1;
    return (u + 2 < Uprime) ? (u + 2) : Uprime-1;
}

void CTCLayer::initWeights() {
    // 2 weights - matrices: one for the forwardHiddenLayer and the other for the backward
    for(int i = 0; i < 2; ++i)
    {
        MatrixXd m = initRandomMatrix(K, H);
        w.push_back(m);
    }
}

MatrixXd CTCLayer::initRandomMatrix(int m, int n) {
    MatrixXd mat(m, n);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution (0.0, 0.1);

//    cout << "weightsMatrix\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            mat(i, j) = distribution(generator);
//            cout << mat(i, j) << " ";
        }
    }
//    cout << "\n";

    return mat;
}

/**
 * Apply exponential function for each element in matrix a
 */
MatrixXd CTCLayer::compute_exp(MatrixXd a) {
    int m = a.rows(), n = a.cols();
    MatrixXd exp_a(m, n);

    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < n; ++j) {
            exp_a(i, j) = safe_exp( a(i,j) );
//            cout << exp_a(i, j) << " ";
        }
//        cout << "\n";
    }

    return exp_a;
}

void CTCLayer::initAlphabet() {
    int k = 0;

    for(int i = 97; i <= 122; ++i, k++) //small letters
        alphabet.insert( pair<char,int>((char)i, k) );

    for(int i = 65; i <= 90; ++i, k++) //capital letters
        alphabet.insert( pair<char,int>((char)i, k) );

    for(int i = 40; i <= 63; ++i, k++) //capital letters
        alphabet.insert( pair<char,int>((char)i, k) );

    alphabet.insert( pair<char,int>('?', k) ); k++;

    alphabet.insert( pair<char,int>('"', k) ); k++;

    alphabet.insert( pair<char,int>(' ', k) ); k++; //white-space
    //TODO: add more characters in the alphabet -- punctuation marks


    map<char,int>::iterator it;

    if(DEBUG)
        for (it=alphabet.begin(); it!=alphabet.end(); ++it)
           cout << it->first << " => " << it->second << '\n';
//    l = "mara";
//    createExtendedLabel();
//    cout << "=" << l_prime << "=\n";
}

void CTCLayer::createExtendedLabel() {
    string aux = " ";
    for(int i = 0; i < l.size(); ++i)
    {
        aux += string(1, l[i]);
        aux += string(1, ' ');
    }
    l_prime = aux;
    cout << "lprime = " << l_prime <<"\n";
}

