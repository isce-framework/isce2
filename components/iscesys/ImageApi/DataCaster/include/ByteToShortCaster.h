#ifndef ByteToShortCaster_h
#define ByteToShortCaster_h

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

class ByteToShortCaster : public DataCaster
{
    public:
        ByteToShortCaster()
        {
            DataSizeIn = sizeof(char);
            DataSizeOut = sizeof(short);
            TCaster = (void *) new Caster<char,short>();
        }
        virtual ~ByteToShortCaster()
        {
            delete (Caster<char,short> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<char,short> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //ByteToShortCaster_h
