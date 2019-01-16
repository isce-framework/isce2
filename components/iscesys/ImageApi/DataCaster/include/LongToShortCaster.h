#ifndef LongToShortCaster_h
#define LongToShortCaster_h

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

class LongToShortCaster : public DataCaster
{
    public:
        LongToShortCaster()
        {
            DataSizeIn = sizeof(long);
            DataSizeOut = sizeof(short);
            TCaster = (void *) new Caster<long,short>();
        }
        virtual ~LongToShortCaster()
        {
            delete (Caster<long,short> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<long,short> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //LongToShortCaster_h
