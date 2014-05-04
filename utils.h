#ifndef UTILS_H
#define UTILS_H

#include <math.h>
#define DEBUG 0
#define DEBUG_GRADIENT 1
#define MIU 0.9

double f(double);
double g(double);
double h(double);
double sigmoid(double);
double tanh_derived(double x);
double hyperbolicTangent(double x);
double f_derived(double x);
double g_derived(double x);
double h_derived(double x);
void updateOneWeight(double ETA, double &w, double &delta_w, double gradient);

#endif // UTILS_H
