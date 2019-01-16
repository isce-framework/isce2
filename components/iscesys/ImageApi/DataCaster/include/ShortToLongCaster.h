#ifndef ShortToLongCaster_h
#define ShortToLongCaster_h

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

class ShortToLongCaster : public DataCaster
{
    public:
        ShortToLongCaster()
        {
            DataSizeIn = sizeof(short);
            DataSizeOut = sizeof(long);
            TCaster = (void *) new Caster<short,long>();
        }
        virtual ~ShortToLongCaster()
        {
            delete (Caster<short,long> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<short,long> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //ShortToLongCaster_h
