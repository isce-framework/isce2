#ifndef IntToFloatCaster_h
#define IntToFloatCaster_h

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

class IntToFloatCaster : public DataCaster
{
    public:
        IntToFloatCaster()
        {
            DataSizeIn = sizeof(int);
            DataSizeOut = sizeof(float);
            TCaster = (void *) new Caster<int,float>();
        }
        virtual ~IntToFloatCaster()
        {
            delete (Caster<int,float> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<int,float> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //IntToFloatCaster_h
