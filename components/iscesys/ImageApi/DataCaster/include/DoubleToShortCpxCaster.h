#ifndef DoubleToShortCpxCaster_h
#define DoubleToShortCpxCaster_h

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

class DoubleToShortCpxCaster : public DataCaster
{
    public:
        DoubleToShortCpxCaster()
        {
            DataSizeIn = sizeof(complex<double>);
            DataSizeOut = sizeof(complex<short>);
            TCaster = (void *) new CasterComplexRound<double,short>();
        }
        virtual ~DoubleToShortCpxCaster()
        {
            delete (CasterComplexRound<double,short> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((CasterComplexRound<double,short> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //DoubleToShortCpxCaster_h
