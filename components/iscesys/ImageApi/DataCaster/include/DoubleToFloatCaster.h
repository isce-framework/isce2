#ifndef DoubleToFloatCaster_h
#define DoubleToFloatCaster_h

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

class DoubleToFloatCaster : public DataCaster
{
    public:
        DoubleToFloatCaster()
        {
            DataSizeIn = sizeof(double);
            DataSizeOut = sizeof(float);
            TCaster = (void *) new Caster<double,float>();
        }
        virtual ~DoubleToFloatCaster()
        {
            delete (Caster<double,float> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((Caster<double,float> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //DoubleToFloatCaster_h
