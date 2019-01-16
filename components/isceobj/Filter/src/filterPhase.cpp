#include <complex>
#include "Image.hh"
#include "MeanFilter.hh"
#include "GaussianFilter.hh"
#include "header.h"

// This is an interface layer between Python and the C++ object creation

int
meanFilterPhase(char *inFile, char *outFile, int imageWidth, int imageHeight, int filterWidth, int filterHeight)
{
  MeanFilter *filter = new MeanFilter(filterWidth,filterHeight);
  Image<std::complex<float> > *inImage   = new Image<std::complex<float> >(inFile,"r",imageWidth,imageHeight);
  Image<std::complex<float> > *outImage = new Image<std::complex<float> >(outFile,"w",imageWidth,imageHeight);

  cpxPhaseFilter(inImage,outImage,filter);

  delete filter;
  delete inImage;
  delete outImage;

  return 1;
}

int
gaussianFilterPhase(char *inFile, char *outFile, int imageWidth, int imageHeight, int filterWidth, int filterHeight, double sigma)
{
  GaussianFilter *filter = new GaussianFilter(filterWidth,filterHeight,sigma);
  Image<std::complex<float> > *inImage   = new Image<std::complex<float> >(inFile,"r",imageWidth,imageHeight);
  Image<std::complex<float> > *outImage = new Image<std::complex<float> >(outFile,"w",imageWidth,imageHeight);

  cpxPhaseFilter(inImage,outImage,filter);

  delete filter;
  delete inImage;
  delete outImage;

  return 1;
}

int
medianFilterPhase(char *inFile, char *outFile, int imageWidth, int imageHeight, int filterWidth, int filterHeight)
{
  Image<std::complex<float> > *inImage   = new Image<std::complex<float> >(inFile,"r",imageWidth,imageHeight);
  Image<std::complex<float> > *outImage = new Image<std::complex<float> >(outFile,"w",imageWidth,imageHeight);

  medianPhaseFilter(inImage,outImage,filterWidth,filterHeight);

  delete inImage;
  delete outImage;

  return 1;
}
