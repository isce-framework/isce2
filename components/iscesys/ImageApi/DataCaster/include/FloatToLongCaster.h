#ifndef FloatToLongCaster_h
#define FloatToLongCaster_h

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

class FloatToLongCaster : public DataCaster
{
    public:
        FloatToLongCaster()
        {
            DataSizeIn = sizeof(float);
            DataSizeOut = sizeof(long);
            TCaster = (void *) new CasterRound<float,long>();
        }
        virtual ~FloatToLongCaster()
        {
            delete (CasterRound<float,long> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((CasterRound<float,long> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //FloatToLongCaster_h
