#ifndef IntToFloatCpxCaster_h
#define IntToFloatCpxCaster_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif
#include <complex>
#include <stdint.h>
#include "DataCaster.h"
#include "CasterComplexInt.h"

using namespace std;

class IntToFloatCpxCaster : public DataCaster
{
    public:
        IntToFloatCpxCaster()
        {
            DataSizeIn = sizeof(complex<int>);
            DataSizeOut = sizeof(complex<float>);
            TCaster = (void *) new CasterComplexInt<int,float>();
        }
        virtual ~IntToFloatCpxCaster()
        {
            delete (CasterComplexInt<int,float> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((CasterComplexInt<int,float> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //IntToFloatCpxCaster_h
