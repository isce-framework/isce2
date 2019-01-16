#include <cmath>
#include "RangePolyInterpolator.h"

void RangePolyInterpolator::getField(double row, double col)
{
    double res;

    res = evalPoly1d(&poly, col);
    return res;
}
