#include <cmath>
#include "Poly1dInterpolator.h"

void
Poly1dInterpolator::getData(char * buf, int row, int col, int & numEl)
{
  if (row < this->NumberOfLines)
  {
    double res;
    for (int i = 0; i < numEl; ++i)
    {
      res = evalPoly1d(poly, (double) col);

      (*(double *) &buf[i * SizeV]) = res;
      ++col;
      //not that here row stand for the dimension that is changing
      if(col == this->LineWidth)
      {
        break;
      }
    }
  }
  else
  {
    NoGoodFlag = 1;
    EofFlag = -1;
  }
  return;
}
void
Poly1dInterpolator::init(void * poly)
{
  this->poly = static_cast<cPoly1d *>(poly);
}
