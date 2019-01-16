#ifndef LongToFloatCaster_h
#define LongToFloatCaster_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif
#include <stdint.h>
#include "DataCaster.h"
#include "Caster.h"

using namespace std;

class LongToFloatCaster : public DataCaster
{
    public:
        LongToFloatCaster()
        {
            DataSizeIn = sizeof(long);
            DataSizeOut = sizeof(float);
            TCaster = (void *) new Caster<long,float>();
        }
        virtual ~LongToFloatCaster()
        {
            delete (Caster<long,float> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<long,float> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //LongToFloatCaster_h
