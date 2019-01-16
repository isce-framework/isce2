#ifndef ShortToIntCpxCaster_h
#define ShortToIntCpxCaster_h

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

class ShortToIntCpxCaster : public DataCaster
{
    public:
        ShortToIntCpxCaster()
        {
            DataSizeIn = sizeof(complex<short>);
            DataSizeOut = sizeof(complex<int>);
            TCaster = (void *) new Caster<complex<short>,complex<int> >();
        }
        virtual ~ShortToIntCpxCaster()
        {
            delete (Caster<complex<short>,complex<int> > *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<complex<short>,complex<int> > *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //ShortToIntCpxCaster_h
