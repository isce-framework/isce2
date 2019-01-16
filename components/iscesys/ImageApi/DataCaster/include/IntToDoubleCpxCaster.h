#ifndef IntToDoubleCpxCaster_h
#define IntToDoubleCpxCaster_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif
#include <complex>
#include <stdint.h>
#include "DataCaster.h"
#include "CasterComplexInt.h"

using namespace std;

class IntToDoubleCpxCaster : public DataCaster
{
    public:
        IntToDoubleCpxCaster()
        {
            DataSizeIn = sizeof(complex<int>);
            DataSizeOut = sizeof(complex<double>);
            TCaster = (void *) new CasterComplexInt<int,double>();
        }
        virtual ~IntToDoubleCpxCaster()
        {
            delete (CasterComplexInt<int,double> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((CasterComplexInt<int,double> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //IntToDoubleCpxCaster_h
