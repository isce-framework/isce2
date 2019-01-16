#include <cmath>
#include "EdgeFilter.hh"

EdgeFilter::EdgeFilter(int width, int height) : Filter(width,height)
{
  if ((this->width%2 == 0) || (this->height%2 == 0))
    {
     throw "Edge Filter dimensions must be odd\n";
    }
  this->setup();
}

void
EdgeFilter::setup()
{
  int x;
  int halfX, halfY;
  double sum = 0.0;

  halfX = floor(width/2);
  halfY = floor(height/2);

  // Construct an Edge Filter
  for(x=0;x<halfX;x++)
    {
      this->setValue(x,halfY,-1.0);
      sum += -1.0;
    }
  this->setValue(halfX,halfY,-sum);

  this->setScale(1.0);
  this->setOffset(0.0);
}
