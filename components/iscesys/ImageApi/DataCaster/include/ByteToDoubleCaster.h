#ifndef ByteToDoubleCaster_h
#define ByteToDoubleCaster_h

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

class ByteToDoubleCaster : public DataCaster
{
    public:
        ByteToDoubleCaster()
        {
            DataSizeIn = sizeof(char);
            DataSizeOut = sizeof(double);
            TCaster = (void *) new Caster<char,double>();
        }
        virtual ~ByteToDoubleCaster()
        {
            delete (Caster<char,double> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<char,double> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //ByteToDoubleCaster_h
