#ifndef ByteToFloatCaster_h
#define ByteToFloatCaster_h

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

class ByteToFloatCaster : public DataCaster
{
    public:
        ByteToFloatCaster()
        {
            DataSizeIn = sizeof(char);
            DataSizeOut = sizeof(float);
            TCaster = (void *) new Caster<char,float>();
        }
        virtual ~ByteToFloatCaster()
        {
            delete (Caster<char,float> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<char,float> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //ByteToFloatCaster_h
