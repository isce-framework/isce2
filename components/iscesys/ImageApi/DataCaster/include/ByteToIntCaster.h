#ifndef ByteToIntCaster_h
#define ByteToIntCaster_h

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

class ByteToIntCaster : public DataCaster
{
    public:
        ByteToIntCaster()
        {
            DataSizeIn = sizeof(char);
            DataSizeOut = sizeof(int);
            TCaster = (void *) new Caster<char,int>();
        }
        virtual ~ByteToIntCaster()
        {
            delete (Caster<char,int> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<char,int> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //ByteToIntCaster_h
