#ifndef FloatToShortCaster_h
#define FloatToShortCaster_h

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

class FloatToShortCaster : public DataCaster
{
    public:
        FloatToShortCaster()
        {
            DataSizeIn = sizeof(float);
            DataSizeOut = sizeof(short);
            TCaster = (void *) new CasterRound<float,short>();
        }
        virtual ~FloatToShortCaster()
        {
            delete (CasterRound<float,short> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((CasterRound<float,short> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //FloatToShortCaster_h
