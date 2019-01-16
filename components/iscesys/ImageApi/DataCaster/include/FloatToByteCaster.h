#ifndef FloatToByteCaster_h
#define FloatToByteCaster_h

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

class FloatToByteCaster : public DataCaster
{
    public:
        FloatToByteCaster()
        {
            DataSizeIn = sizeof(float);
            DataSizeOut = sizeof(char);
            TCaster = (void *) new CasterRound<float,char>();
        }
        virtual ~FloatToByteCaster()
        {
            delete (CasterRound<float,char> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((CasterRound<float,char> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //FloatToByteCaster_h
