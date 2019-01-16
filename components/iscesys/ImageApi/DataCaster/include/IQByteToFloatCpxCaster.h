#ifndef IQByteToFloatCpxCaster_h
#define IQByteToFloatCpxCaster_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif
#include <stdint.h>
#include "DataCaster.h"
#include <complex>

using namespace std;

//If we need more that a datatype out the class can use template like the one derived form the
//Caster class
class IQByteToFloatCpxCaster : public DataCaster
{
public:
  IQByteToFloatCpxCaster()
  {
    DataSizeIn = sizeof(complex<char>);
    DataSizeOut = sizeof(complex<float>);
    mask = 255;
  }
  virtual
  ~IQByteToFloatCpxCaster()
  {
  }
  void
  convert(char * in, char * out, int numEl)
  {
    for (int i = 0, j = 0, k = 0; i < numEl; ++i, j += this->DataSizeIn, k +=
        this->DataSizeOut)
    {
      //this is if datatype is short. if it's byte need to change
      int val1 = (mask & in[j+iqflip]);
      int val2 = (mask & in[j+1-iqflip]);

      (*(complex<float> *) &out[k]) =  complex<float>(val1-xmi,val2-xmq);
    }
  }

  void
  setXmi(float xmi)
  {
    this->xmi = xmi;
  }

  void
  setXmq(float xmq)
  {
    this->xmq = xmq;
  }

  void
  setIQflip(int iqflip)
  {
    this->iqflip = iqflip;
  }

private:
  float xmi;
  float xmq;
  int iqflip;
  uint8_t mask;
};
#endif //IQByteToFloatCpxCaster_h
