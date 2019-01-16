#include <complex>
#include "Image.hh"
#include "Filter.hh"

void
cpxPhaseFilter(Image<std::complex<float> > *image, Image<std::complex<float> > *result, Filter *filter)
{
  int x,y,filterX,filterY;
  int imageWidth = image->getWidth();
  int imageHeight = image->getHeight();
  int w = imageWidth;
  int h = imageHeight;

#pragma omp parallel for private(x,y,filterX,filterY) shared(image,filter,result)
  for (x=0;x<w;x++)
    {
      for (y=0;y<h;y++)
	{
	  float phase = 0.0;
	  for (filterX=0;filterX<filter->getWidth();filterX++)
	    {
	      for (filterY=0;filterY<filter->getHeight();filterY++)
		{
		  int imageX = (x-filter->getWidth()/2 + filterX + w) % w;
		  int imageY = (y-filter->getHeight()/2 + filterY + h) % h;
		  std::complex<float> cpx = image->getValue(imageX,imageY);
		  phase += arg(cpx) * filter->getValue(filterX,filterY);
		}
	    }
	    float mag = abs(image->getValue(x,y));
	    float arg = filter->getScale()*phase + filter->getOffset();
	    std::complex<float> ans = std::polar(mag,arg);
	    result->setValue(x,y,ans);
	}
    }
}
