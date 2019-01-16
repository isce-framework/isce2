#include "Filter.hh"

Filter::Filter(int width, int height)
{
  if ((width <= 0) || (height <= 0)) { throw "Filter dimensions must be positive";}
  this->width = width;
  this->height = height;
  this->filter = new double[width*height];
}

Filter::~Filter()
{
  delete [] this->filter;
}

void Filter::setWidth(int width) { this->width = width; }
void Filter::setHeight(int height) { this->height = height; }
void Filter::setScale(double scale) { this->scale = scale; }
void Filter::setOffset(double offset) { this->offset = offset; }
void Filter::setValue(int x, int y, double value) { this->filter[y*width + x] = value; }

int Filter::getWidth() { return this->width; }
int Filter::getHeight() { return this->height; }
double Filter::getScale() { return this->scale; }
double Filter::getOffset() { return this->offset; }

double
Filter::getValue(int x, int y)
{
  return this->filter[y*width + x];
}
