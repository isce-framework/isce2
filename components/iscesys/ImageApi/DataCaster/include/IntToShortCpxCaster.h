#ifndef IntToShortCpxCaster_h
#define IntToShortCpxCaster_h

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

class IntToShortCpxCaster : public DataCaster
{
    public:
        IntToShortCpxCaster()
        {
            DataSizeIn = sizeof(complex<int>);
            DataSizeOut = sizeof(complex<short>);
            TCaster = (void *) new Caster<complex<int>,complex<short> >();
        }
        virtual ~IntToShortCpxCaster()
        {
            delete (Caster<complex<int>,complex<short> > *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<complex<int>,complex<short> > *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //IntToShortCpxCaster_h
