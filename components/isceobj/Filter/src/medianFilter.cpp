#include <cmath>
#include <cstdlib>
#include <complex>
#include "Image.hh"

int compare(const void *a, const void *b);

void
medianPhaseFilter(Image<std::complex<float> > *image, Image<std::complex<float> > *result, int filterWidth, int filterHeight)
{
  int x,y,filterX,filterY;
  int imageWidth = image->getWidth();
  int imageHeight = image->getHeight();
  int w = imageWidth;
  int h = imageHeight;
  float phase[filterWidth*filterHeight];
  
#pragma omp parallel for private(x,y,filterX,filterY,phase) shared(image,result)
  for (x=0;x<w;x++)
    {
      for (y=0;y<h;y++)
	{
	  int n = 0;
	  for (filterX=0;filterX<filterWidth;filterX++)
	    {
	      for (filterY=0;filterY<filterHeight;filterY++)
		{
		  int imageX = (x-filterWidth/2 + filterX + w) % w;
		  int imageY = (y-filterHeight/2 + filterY + h) % h;
		  std::complex<float> cpx = image->getValue(imageX,imageY);
		  phase[n] = arg(cpx);
		  n++;
		}
	    }
	  
	  //heapsort(phase, filterWidth*filterHeight, sizeof(float), compare);
	  qsort(phase, filterWidth*filterHeight, sizeof(float), compare);
	  float carg;

	  // Calculate the median
	  if ((filterWidth*filterHeight) % 2 == 1)
	    {
	      carg = phase[filterWidth*filterHeight/2];
	    }
	  else if (filterWidth >= 2)
	    {
	      carg = (phase[filterWidth*filterHeight/2] + phase[filterWidth*filterHeight/2 + 1])/2;
	    }
	  float mag = abs(image->getValue(x,y));
	  std::complex<float> ans = std::polar(mag,carg);
	  result->setValue(x,y,ans);
	}
    }
}


int
compare(const void *a, const void *b)
{
  // First, convert the void pointer to a pointer of known type
  const float *fa = (const float *)a;
  const float *fb = (const float *)b;
  
  // Then, dereference the pointers and return their difference
  // if this difference is negative, then b is larger than a
  // if this differene is positive, then a is larger than b
  return (int)(*fa - *fb);
}
