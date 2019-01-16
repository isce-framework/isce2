#ifndef IntToLongCaster_h
#define IntToLongCaster_h

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

class IntToLongCaster : public DataCaster
{
    public:
        IntToLongCaster()
        {
            DataSizeIn = sizeof(int);
            DataSizeOut = sizeof(long);
            TCaster = (void *) new Caster<int,long>();
        }
        virtual ~IntToLongCaster()
        {
            delete (Caster<int,long> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<int,long> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //IntToLongCaster_h
