#ifndef LongToDoubleCaster_h
#define LongToDoubleCaster_h

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

class LongToDoubleCaster : public DataCaster
{
    public:
        LongToDoubleCaster()
        {
            DataSizeIn = sizeof(long);
            DataSizeOut = sizeof(double);
            TCaster = (void *) new Caster<long,double>();
        }
        virtual ~LongToDoubleCaster()
        {
            delete (Caster<long,double> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<long,double> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //LongToDoubleCaster_h
