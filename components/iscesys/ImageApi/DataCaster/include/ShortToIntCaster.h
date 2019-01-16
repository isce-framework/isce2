#ifndef ShortToIntCaster_h
#define ShortToIntCaster_h

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

class ShortToIntCaster : public DataCaster
{
    public:
        ShortToIntCaster()
        {
            DataSizeIn = sizeof(short);
            DataSizeOut = sizeof(int);
            TCaster = (void *) new Caster<short,int>();
        }
        virtual ~ShortToIntCaster()
        {
            delete (Caster<short,int> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<short,int> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //ShortToIntCaster_h
