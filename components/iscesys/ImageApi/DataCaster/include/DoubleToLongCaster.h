#ifndef DoubleToLongCaster_h
#define DoubleToLongCaster_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif
#include <stdint.h>
#include "DataCaster.h"
#include "CasterRound.h"

using namespace std;

class DoubleToLongCaster : public DataCaster
{
    public:
        DoubleToLongCaster()
        {
            DataSizeIn = sizeof(double);
            DataSizeOut = sizeof(long);
            TCaster = (void *) new CasterRound<double,long>();
        }
        virtual ~DoubleToLongCaster()
        {
            delete (CasterRound<double,long> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((CasterRound<double,long> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //DoubleToLongCaster_h
