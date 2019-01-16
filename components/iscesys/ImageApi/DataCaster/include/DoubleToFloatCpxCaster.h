#ifndef DoubleToFloatCpxCaster_h
#define DoubleToFloatCpxCaster_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif
#include <complex>
#include <stdint.h>
#include "DataCaster.h"
#include "Caster.h"

using namespace std;

class DoubleToFloatCpxCaster : public DataCaster
{
    public:
        DoubleToFloatCpxCaster()
        {
            DataSizeIn = sizeof(complex<double>);
            DataSizeOut = sizeof(complex<float>);
            TCaster = (void *) new Caster<complex<double>,complex<float> >();
        }
        virtual ~DoubleToFloatCpxCaster()
        {
            delete (Caster<complex<double>,complex<float> > *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<complex<double>,complex<float> > *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //DoubleToFloatCpxCaster_h
