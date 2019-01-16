#ifndef LongToIntCaster_h
#define LongToIntCaster_h

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

class LongToIntCaster : public DataCaster
{
    public:
        LongToIntCaster()
        {
            DataSizeIn = sizeof(long);
            DataSizeOut = sizeof(int);
            TCaster = (void *) new Caster<long,int>();
        }
        virtual ~LongToIntCaster()
        {
            delete (Caster<long,int> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<long,int> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //LongToIntCaster_h
