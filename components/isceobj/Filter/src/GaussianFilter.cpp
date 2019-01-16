#include <iostream>
#include <cmath>
#include "GaussianFilter.hh"

GaussianFilter::GaussianFilter(int width, int height) : Filter(width,height)
{
 this->sigma2 = 1.0;
 this->setup();
}

GaussianFilter::GaussianFilter(int width, int height, double sigma2) : Filter(width,height)
{
  this->sigma2 = sigma2;
  this->setup();
}

void
GaussianFilter::setup()
{
  int x,y;
  double sum = 0.0;

  for(x=0;x<this->width;x++)
    {
      double filterX = (x-floor(this->width/2.0));
      for(y=0;y<this->height;y++)
	{
	  double filterY = (floor(this->height/2.0)-y);
	  double val = this->G(filterX,filterY);
	  sum += val;
	  this->setValue(x,y,val);
	}

    }
  this->setScale(1.0/sum);
  this->setOffset(0.0);
}

double
GaussianFilter::G(double x, double y)
{
 return exp(-(x*x + y*y)/(2.0*this->sigma2))/(2.0*M_PI*this->sigma2);
}
