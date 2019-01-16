#include <cmath>
#include "AzimuthPolyInterpolator.h"

void AzimuthPolyInterpolator::getField(double row, double col)
{
    double res;
    res = evalPoly1d(&poly, row);
    return res;
}
