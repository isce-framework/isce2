#include "Filter.hh"

class EdgeFilter: public Filter
{
 private:
   void setup();
 public:
   EdgeFilter(int width, int height);
};
