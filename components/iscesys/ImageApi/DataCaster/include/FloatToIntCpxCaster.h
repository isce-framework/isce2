#ifndef FloatToIntCpxCaster_h
#define FloatToIntCpxCaster_h

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

class FloatToIntCpxCaster : public DataCaster
{
    public:
        FloatToIntCpxCaster()
        {
            DataSizeIn = sizeof(complex<float>);
            DataSizeOut = sizeof(complex<int>);
            TCaster = (void *) new CasterComplexRound<float,int>();
        }
        virtual ~FloatToIntCpxCaster()
        {
            delete (CasterComplexRound<float,int> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((CasterComplexRound<float,int> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //FloatToIntCpxCaster_h
