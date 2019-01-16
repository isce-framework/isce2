#ifndef FloatToShortCpxCaster_h
#define FloatToShortCpxCaster_h

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

class FloatToShortCpxCaster : public DataCaster
{
    public:
        FloatToShortCpxCaster()
        {
            DataSizeIn = sizeof(complex<float>);
            DataSizeOut = sizeof(complex<short>);
            TCaster = (void *) new CasterComplexRound<float,short>();
        }
        virtual ~FloatToShortCpxCaster()
        {
            delete (CasterComplexRound<float,short> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((CasterComplexRound<float,short> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //FloatToShortCpxCaster_h
