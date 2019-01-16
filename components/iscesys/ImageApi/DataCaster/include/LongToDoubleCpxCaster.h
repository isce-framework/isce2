#ifndef LongToDoubleCpxCaster_h
#define LongToDoubleCpxCaster_h

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

class LongToDoubleCpxCaster : public DataCaster
{
    public:
        LongToDoubleCpxCaster()
        {
            DataSizeIn = sizeof(complex<long>);
            DataSizeOut = sizeof(complex<double>);
            TCaster = (void *) new CasterComplexInt<long,double>();
        }
        virtual ~LongToDoubleCpxCaster()
        {
            delete (CasterComplexInt<long,double> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((CasterComplexInt<long,double> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //LongToDoubleCpxCaster_h
