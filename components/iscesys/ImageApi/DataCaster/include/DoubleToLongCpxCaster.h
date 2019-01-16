#ifndef DoubleToLongCpxCaster_h
#define DoubleToLongCpxCaster_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif
#include <complex>
#include <stdint.h>
#include "DataCaster.h"
#include "CasterComplexRound.h"

using namespace std;

class DoubleToLongCpxCaster : public DataCaster
{
    public:
        DoubleToLongCpxCaster()
        {
            DataSizeIn = sizeof(complex<double>);
            DataSizeOut = sizeof(complex<long>);
            TCaster = (void *) new CasterComplexRound<double,long>();
        }
        virtual ~DoubleToLongCpxCaster()
        {
            delete (CasterComplexRound<double,long> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((CasterComplexRound<double,long> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //DoubleToLongCpxCaster_h
