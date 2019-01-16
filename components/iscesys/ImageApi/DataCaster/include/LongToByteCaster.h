#ifndef LongToByteCaster_h
#define LongToByteCaster_h

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

class LongToByteCaster : public DataCaster
{
    public:
        LongToByteCaster()
        {
            DataSizeIn = sizeof(long);
            DataSizeOut = sizeof(char);
            TCaster = (void *) new Caster<long,char>();
        }
        virtual ~LongToByteCaster()
        {
            delete (Caster<long,char> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<long,char> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //LongToByteCaster_h
