#ifndef ShortToDoubleCpxCaster_h
#define ShortToDoubleCpxCaster_h

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

class ShortToDoubleCpxCaster : public DataCaster
{
    public:
        ShortToDoubleCpxCaster()
        {
            DataSizeIn = sizeof(complex<short>);
            DataSizeOut = sizeof(complex<double>);
            TCaster = (void *) new CasterComplexInt<short,double>();
        }
        virtual ~ShortToDoubleCpxCaster()
        {
            delete (CasterComplexInt<short,double> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((CasterComplexInt<short,double> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //ShortToDoubleCpxCaster_h
