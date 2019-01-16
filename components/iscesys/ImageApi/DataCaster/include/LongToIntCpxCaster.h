#ifndef LongToIntCpxCaster_h
#define LongToIntCpxCaster_h

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

class LongToIntCpxCaster : public DataCaster
{
    public:
        LongToIntCpxCaster()
        {
            DataSizeIn = sizeof(complex<long>);
            DataSizeOut = sizeof(complex<int>);
            TCaster = (void *) new Caster<complex<long>,complex<int> >();
        }
        virtual ~LongToIntCpxCaster()
        {
            delete (Caster<complex<long>,complex<int> > *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<complex<long>,complex<int> > *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //LongToIntCpxCaster_h
