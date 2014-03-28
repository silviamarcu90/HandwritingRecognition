#include "utils.h"

double f(double x) {
    return sigmoid(x);
}

double g(double x) {
    return //sigmoid(x);//
    tanh(x);
}

double h(double x) {
    return //sigmoid(x);//
    tanh(x);
}

double sigmoid(double x) {
    if(x >= 0) return 1/(1 + exp(-x));
    else return exp(x)/(1 + exp(x));
}

double f_derived(double x) {
    return f(x)*(1 - f(x));
}

//TODO -- take the derivative of the tanh function
double h_derived(double x) {
    return f(x)*(1 - f(x));
}

double g_derived(double x) {
    return h_derived(x);
}
