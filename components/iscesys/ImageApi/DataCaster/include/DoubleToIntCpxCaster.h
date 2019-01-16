#ifndef DoubleToIntCpxCaster_h
#define DoubleToIntCpxCaster_h

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

class DoubleToIntCpxCaster : public DataCaster
{
    public:
        DoubleToIntCpxCaster()
        {
            DataSizeIn = sizeof(complex<double>);
            DataSizeOut = sizeof(complex<int>);
            TCaster = (void *) new CasterComplexRound<double,int>();
        }
        virtual ~DoubleToIntCpxCaster()
        {
            delete (CasterComplexRound<double,int> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((CasterComplexRound<double,int> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //DoubleToIntCpxCaster_h
