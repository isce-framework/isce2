#ifndef ShortToByteCaster_h
#define ShortToByteCaster_h

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

class ShortToByteCaster : public DataCaster
{
    public:
        ShortToByteCaster()
        {
            DataSizeIn = sizeof(short);
            DataSizeOut = sizeof(char);
            TCaster = (void *) new Caster<short,char>();
        }
        virtual ~ShortToByteCaster()
        {
            delete (Caster<short,char> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<short,char> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //ShortToByteCaster_h
