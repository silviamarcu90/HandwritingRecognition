#ifndef LOG_H
#define LOG_H

#include <math.h>

static const double expMax = std::numeric_limits<double>::max();
static const double expMin = std::numeric_limits<double>::min();
static const double expLimit = std::log(expMax);
static const double logInfinity = 1e100;
static const double logZero = -logInfinity;

//static functions
static double safe_exp(double x)
{
    if (x == logZero)
    {
        return 0;
    }
    if (x >= expLimit)
    {
        return expMax;
    }
    return std::exp(x);
}
static double safe_log(double x)
{
    if (x <= expMin)
    {
        return logZero;
    }
    return std::log(x);
}
static double log_add(double x, double y)
{
    if (x == logZero)
    {
        return y;
    }
    if (y == logZero)
    {
        return x;
    }
    if (x < y)
    {
        std::swap(x, y);
    }
    return x + std::log(1.0 + safe_exp(y - x));
}

static double log_multiply(double x, double y)
{
    if (x == logZero || y == logZero)
    {
        return logZero;
    }
    return x + y;
}
static double log_divide(double x, double y)
{
    if (x == logZero)
    {
        return logZero;
    }
    if (y == logZero)
    {
        return logInfinity;
    }
    return x - y;
}

#endif // LOG_H
