#include "Filter.hh"

class GaussianFilter: public Filter
{
 private:
   double sigma2;
   void setup();
   double G(double x, double y);
 public:
   GaussianFilter(int width, int height);
   GaussianFilter(int width, int height,double sigma2);
};
