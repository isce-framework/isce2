#ifndef CasterComplexRound_h
#define CasterComplexRound_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif
#include <stdint.h>
#include "DataCasterT.h"
#include <math.h>
#include <complex>

using namespace std;
template<typename F, typename T>

  class CasterComplexRound: public DataCasterT<F,T>
  {
  public:

      CasterComplexRound()
      {
        this->DataSizeIn = sizeof(complex<F>);
        this->DataSizeOut = sizeof(complex<T>);
      }

    virtual ~CasterComplexRound()
    {}
    void
    convert(char * in, char * out, int numEl)
    {
      for (int i = 0, j = 0, k = 0; i < numEl; ++i, j += this->DataSizeIn, k +=
          this->DataSizeOut)
      {
        complex<F> * tmp = (complex<F> *) &in[j];
        (*(complex<T> *) &out[k]) = (complex<T>) complex<double>(round((double) real((*tmp))),round((double) imag((*tmp))));
      }
    }

  };
#endif //CasterComplexRound_h
