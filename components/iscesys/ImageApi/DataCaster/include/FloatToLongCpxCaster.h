#ifndef FloatToLongCpxCaster_h
#define FloatToLongCpxCaster_h

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

class FloatToLongCpxCaster : public DataCaster
{
    public:
        FloatToLongCpxCaster()
        {
            DataSizeIn = sizeof(complex<float>);
            DataSizeOut = sizeof(complex<long>);
            TCaster = (void *) new CasterComplexRound<float,long>();
        }
        virtual ~FloatToLongCpxCaster()
        {
            delete (CasterComplexRound<float,long> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((CasterComplexRound<float,long> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //FloatToLongCpxCaster_h
