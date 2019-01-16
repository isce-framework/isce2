#ifndef ShortToFloatCpxCaster_h
#define ShortToFloatCpxCaster_h

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

class ShortToFloatCpxCaster : public DataCaster
{
    public:
        ShortToFloatCpxCaster()
        {
            DataSizeIn = sizeof(complex<short>);
            DataSizeOut = sizeof(complex<float>);
            TCaster = (void *) new CasterComplexInt<short,float>();
        }
        virtual ~ShortToFloatCpxCaster()
        {
            delete (CasterComplexInt<short,float> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((CasterComplexInt<short,float> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //ShortToFloatCpxCaster_h
