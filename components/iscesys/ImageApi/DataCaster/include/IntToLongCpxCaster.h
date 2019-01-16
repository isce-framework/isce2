#ifndef IntToLongCpxCaster_h
#define IntToLongCpxCaster_h

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

class IntToLongCpxCaster : public DataCaster
{
    public:
        IntToLongCpxCaster()
        {
            DataSizeIn = sizeof(complex<int>);
            DataSizeOut = sizeof(complex<long>);
            TCaster = (void *) new Caster<complex<int>,complex<long> >();
        }
        virtual ~IntToLongCpxCaster()
        {
            delete (Caster<complex<int>,complex<long> > *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<complex<int>,complex<long> > *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //IntToLongCpxCaster_h
