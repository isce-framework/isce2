#ifndef IntToShortCaster_h
#define IntToShortCaster_h

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

class IntToShortCaster : public DataCaster
{
    public:
        IntToShortCaster()
        {
            DataSizeIn = sizeof(int);
            DataSizeOut = sizeof(short);
            TCaster = (void *) new Caster<int,short>();
        }
        virtual ~IntToShortCaster()
        {
            delete (Caster<int,short> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<int,short> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //IntToShortCaster_h
