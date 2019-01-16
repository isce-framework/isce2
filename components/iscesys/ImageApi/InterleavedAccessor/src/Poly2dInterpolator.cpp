#include <cmath>
#include "Poly2dInterpolator.h"

void
Poly2dInterpolator::getData(char * buf, int row, int col, int & numEl)
{

  if (row < this->NumberOfLines && col < this->LineWidth)
  {
    double res;

    for (int i = 0; i < numEl; ++i)
    {
      res = evalPoly2d(poly, (double) row, (double)col);

      (*(double *) &buf[i * SizeV]) = res;
      col++;
      if(col == this->LineWidth)
      {
        col = 0;
        row++;
      }
      if (row == this->NumberOfLines)
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
Poly2dInterpolator::init(void * poly)
{
  this->poly = static_cast<cPoly2d *>(poly);
}
