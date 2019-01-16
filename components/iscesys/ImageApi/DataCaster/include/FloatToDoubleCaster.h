#ifndef FloatToDoubleCaster_h
#define FloatToDoubleCaster_h

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

class FloatToDoubleCaster : public DataCaster
{
    public:
        FloatToDoubleCaster()
        {
            DataSizeIn = sizeof(float);
            DataSizeOut = sizeof(double);
            TCaster = (void *) new Caster<float,double>();
        }
        virtual ~FloatToDoubleCaster()
        {
            delete (Caster<float,double> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<float,double> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //FloatToDoubleCaster_h
