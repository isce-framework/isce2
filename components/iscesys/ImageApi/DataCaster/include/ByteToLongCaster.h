#ifndef ByteToLongCaster_h
#define ByteToLongCaster_h

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

class ByteToLongCaster : public DataCaster
{
    public:
        ByteToLongCaster()
        {
            DataSizeIn = sizeof(char);
            DataSizeOut = sizeof(long);
            TCaster = (void *) new Caster<char,long>();
        }
        virtual ~ByteToLongCaster()
        {
            delete (Caster<char,long> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<char,long> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //ByteToLongCaster_h
