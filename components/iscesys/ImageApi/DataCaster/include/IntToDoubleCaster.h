#ifndef IntToDoubleCaster_h
#define IntToDoubleCaster_h

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

class IntToDoubleCaster : public DataCaster
{
    public:
        IntToDoubleCaster()
        {
            DataSizeIn = sizeof(int);
            DataSizeOut = sizeof(double);
            TCaster = (void *) new Caster<int,double>();
        }
        virtual ~IntToDoubleCaster()
        {
            delete (Caster<int,double> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<int,double> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //IntToDoubleCaster_h
