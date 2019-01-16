#ifndef AzimuthPolyInterpolator_h
#define AzimuthPolyInterpolator_h


#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif

#include <cmath>
#include "FieldInterpolator.h"
#include "poly1d.h"

class AzimuthPolyInterpolator: public FieldInterpolator
{
    public:
        AzimuthPolyInterpolator():FieldInterpolator(){}
        virtual ~AzimuthPolyInterpolator(){delete [] Data;}

        double getField(double row, double col);

    protected:
        cPoly1d poly;
};

#endif //AzimuthPolyInterpolator_h
