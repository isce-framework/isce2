#include "MeanFilter.hh"

MeanFilter::MeanFilter(int width, int height) : Filter(width,height)
{
  int x,y;

  // Construct a Mean Filter
  for(x=0;x<this->width;x++)
    {
      for(y=0;y<this->height;y++)
	{
          this->setValue(x,y,1.0);
	}
    }

  this->setScale(1.0/(width*height));
  this->setOffset(0.0);
}
