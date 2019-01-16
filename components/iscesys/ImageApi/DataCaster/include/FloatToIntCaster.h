#ifndef FloatToIntCaster_h
#define FloatToIntCaster_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif
#include <stdint.h>
#include "DataCaster.h"
#include "CasterRound.h"

using namespace std;

class FloatToIntCaster : public DataCaster
{
    public:
        FloatToIntCaster()
        {
            DataSizeIn = sizeof(float);
            DataSizeOut = sizeof(int);
            TCaster = (void *) new CasterRound<float,int>();
        }
        virtual ~FloatToIntCaster()
        {
            delete (CasterRound<float,int> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((CasterRound<float,int> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //FloatToIntCaster_h
