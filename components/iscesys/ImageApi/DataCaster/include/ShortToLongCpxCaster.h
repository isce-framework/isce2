#ifndef ShortToLongCpxCaster_h
#define ShortToLongCpxCaster_h

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

class ShortToLongCpxCaster : public DataCaster
{
    public:
        ShortToLongCpxCaster()
        {
            DataSizeIn = sizeof(complex<short>);
            DataSizeOut = sizeof(complex<long>);
            TCaster = (void *) new Caster<complex<short>,complex<long> >();
        }
        virtual ~ShortToLongCpxCaster()
        {
            delete (Caster<complex<short>,complex<long> > *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<complex<short>,complex<long> > *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //ShortToLongCpxCaster_h
