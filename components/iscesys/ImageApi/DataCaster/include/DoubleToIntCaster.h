#ifndef DoubleToIntCaster_h
#define DoubleToIntCaster_h

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

class DoubleToIntCaster : public DataCaster
{
    public:
        DoubleToIntCaster()
        {
            DataSizeIn = sizeof(double);
            DataSizeOut = sizeof(int);
            TCaster = (void *) new CasterRound<double,int>();
        }
        virtual ~DoubleToIntCaster()
        {
            delete (CasterRound<double,int> *) TCaster;
        }
        void convert(char * in,char * out, int numEl)
        {
            ((CasterRound<double,int> *) (TCaster))->convert(in, out, numEl);
        }

};
#endif //DoubleToIntCaster_h
