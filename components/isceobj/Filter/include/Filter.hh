#ifndef FILTER_HH
#define FILTER_HH 1

class Filter
{
  protected:
   int width;
   int height;
   double scale;
   double offset;
   double *filter;
   void setWidth(int width);
   void setHeight(int height);
   void setValue(int x, int y, double value);
  public:
   Filter(int width, int height);
   ~Filter();
   int getWidth();
   int getHeight();
   double getScale();
   double getOffset();
   double getValue(int x, int y);
   void setScale(double scale);
   void setOffset(double offset);
};

#endif
