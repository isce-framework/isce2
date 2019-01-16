#ifndef Poly1dInterpolator_h
#define Poly1dInterpolator_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif

#include <cmath>
#include "poly1d.h"
#include "InterleavedAccessor.h"
class Poly1dInterpolator : public InterleavedAccessor
{
public:
  Poly1dInterpolator() :
      InterleavedAccessor()
  {
  }
  virtual
  ~Poly1dInterpolator()
  {
  }
  void init(void * poly);


  void
  getData(char * buf, int row, int col, int & numEl);
  //the next functions are pure abstract and need to be implemented, so we just create and empty body
  void
  getDataBand(char *buf, int row, int col, int &numEl, int band){}
  void
  setData(char * buf, int row, int col, int numEl){}
  void
  setDataBand(char * buf, int row, int col, int numEl, int band) {}
protected:
  cPoly1d * poly;
};

#endif //Poly1dInterpolator_h
