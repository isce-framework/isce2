#include <cmath>
#include "Poly2dInterpolator.h"

void Poly2dInterpolator::getData(char * buf, int row, int col, int & numEl);
)
{
    double res;
    for(int i = 0; i < numEl; ++i)
    {
      res = evalPoly2d(poly,row,col);
      (* &buf[i*SizeV]) = res;
    }
    return;
}
void Poly2dInterpolator::init(void * poly)
{
  this.poly = static_cast<cPoly2d *> poly;
}
