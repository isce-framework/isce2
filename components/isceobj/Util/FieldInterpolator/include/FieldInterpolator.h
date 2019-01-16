#ifndef FieldInterpolator_h
#define FieldInterpolator_h


#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif

#include <stdlib.h>
#include <string>
using namespace std;

class FieldInterpolator
{
    public:
        FieldInterpolator(){}
        virtual ~FieldInterpolator(){}

        virtual double getField(double row, double col)=0;

    protected:

};

#endif //Field_interpolator_h
