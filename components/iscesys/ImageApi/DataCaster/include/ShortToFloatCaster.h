#ifndef ShortToFloatCaster_h
#define ShortToFloatCaster_h

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

class ShortToFloatCaster : public DataCaster
{
    public:
        ShortToFloatCaster()
        {
            DataSizeIn = sizeof(short);
            DataSizeOut = sizeof(float);
            TCaster = (void *) new Caster<short,float>();
        }
        virtual ~ShortToFloatCaster()
        {
            delete (Caster<short,float> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<short,float> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //ShortToFloatCaster_h
