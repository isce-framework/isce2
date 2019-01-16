#ifndef FloatToDoubleCpxCaster_h
#define FloatToDoubleCpxCaster_h

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

class FloatToDoubleCpxCaster : public DataCaster
{
    public:
        FloatToDoubleCpxCaster()
        {
            DataSizeIn = sizeof(complex<float>);
            DataSizeOut = sizeof(complex<double>);
            TCaster = (void *) new Caster<complex<float>,complex<double> >();
        }
        virtual ~FloatToDoubleCpxCaster()
        {
            delete (Caster<complex<float>,complex<double> > *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<complex<float>,complex<double> > *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //FloatToDoubleCpxCaster_h
