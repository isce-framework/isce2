#ifndef IntToByteCaster_h
#define IntToByteCaster_h

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

class IntToByteCaster : public DataCaster
{
    public:
        IntToByteCaster()
        {
            DataSizeIn = sizeof(int);
            DataSizeOut = sizeof(char);
            TCaster = (void *) new Caster<int,char>();
        }
        virtual ~IntToByteCaster()
        {
            delete (Caster<int,char> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<int,char> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //IntToByteCaster_h
