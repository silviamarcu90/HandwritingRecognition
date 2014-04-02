#include "utils.h"

double f(double x) {
    return sigmoid(x);
}

double g(double x) {
    return //sigmoid(x);// tanh(x);
            hyperbolicTangent(x);
}

double h(double x) {
    return //sigmoid(x);//tanh(x);
            hyperbolicTangent(x);
}

double sigmoid(double x) {
    if(x >= 0) return 1/(1 + exp(-x)); //avoid overflow
    else return exp(x)/(1 + exp(x));
}

double hyperbolicTangent(double x) {
    if(x >= 0) return (1 - exp(-2*x))/(1 + exp(-2*x)); //avoid overflow
    else return exp(2*x)-1/(exp(2*x) + 1);
}

double f_derived(double x) {
    return f(x)*(1 - f(x));
}

double tanh_derived(double x) {
    double y = hyperbolicTangent(x);
    return 1 - ( y * y );
}

double h_derived(double x) {
    return tanh_derived(x);
}

double g_derived(double x) {
    return tanh_derived(x);
}
