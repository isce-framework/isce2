#ifndef ShortToDoubleCaster_h
#define ShortToDoubleCaster_h

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

class ShortToDoubleCaster : public DataCaster
{
    public:
        ShortToDoubleCaster()
        {
            DataSizeIn = sizeof(short);
            DataSizeOut = sizeof(double);
            TCaster = (void *) new Caster<short,double>();
        }
        virtual ~ShortToDoubleCaster()
        {
            delete (Caster<short,double> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<short,double> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //ShortToDoubleCaster_h
