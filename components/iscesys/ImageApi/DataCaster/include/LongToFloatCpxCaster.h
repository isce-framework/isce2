#ifndef LongToFloatCpxCaster_h
#define LongToFloatCpxCaster_h

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

class LongToFloatCpxCaster : public DataCaster
{
    public:
        LongToFloatCpxCaster()
        {
            DataSizeIn = sizeof(complex<long>);
            DataSizeOut = sizeof(complex<float>);
            TCaster = (void *) new CasterComplexInt<long,float>();
        }
        virtual ~LongToFloatCpxCaster()
        {
            delete (CasterComplexInt<long,float> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((CasterComplexInt<long,float> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //LongToFloatCpxCaster_h
