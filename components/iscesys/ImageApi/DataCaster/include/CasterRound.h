#ifndef CasterRound_h
#define CasterRound_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif
#include <stdint.h>
#include "DataCasterT.h"
#include <math.h>

using namespace std;
template<typename F, typename T>
  class CasterRound : public DataCasterT<F,T>
  {
  public:

      CasterRound()
      {
        this->DataSizeIn = sizeof(F);
        this->DataSizeOut = sizeof(T);
      }

    virtual ~CasterRound()
    {}
    void
    convert(char * in, char * out, int numEl)
    {
      for (int i = 0, j = 0, k = 0; i < numEl; ++i, j += this->DataSizeIn, k +=
          this->DataSizeOut)
      {
        F * tmp = (F *) &in[j];
        (*(T *) &out[k]) = (T) round((double)(*tmp));
      }
    }

  };
#endif //CasterRound_h
