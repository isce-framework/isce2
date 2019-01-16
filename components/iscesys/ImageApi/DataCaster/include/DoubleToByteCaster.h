#ifndef DoubleToByteCaster_h
#define DoubleToByteCaster_h

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

class DoubleToByteCaster : public DataCaster
{
    public:
        DoubleToByteCaster()
        {
            DataSizeIn = sizeof(double);
            DataSizeOut = sizeof(char);
            TCaster = (void *) new CasterRound<double,char>();
        }
        virtual ~DoubleToByteCaster()
        {
            delete (CasterRound<double,char> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((CasterRound<double,char> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //DoubleToByteCaster_h
