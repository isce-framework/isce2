#ifndef DataCaster_h
#define DataCaster_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif

#include <stdint.h>

using namespace std;
/*
 * Caster and CasterRound inherit from DataCasterT which is a template base class for casting.
 * DataCaster is a wrapper for Caster. If the class can be templated then inherit from it
 * and make an actual instantiation of the data type (see DoubleToByteCaster.h for instance).
 * Unfortunately it was not possible to make DataCaster a template.
 * If the class deals with specific datatypes and the template is not used then just inherit from
 * DataCaster and implement convert.
 *
 * How it works: Each XToYCaster inherits from DataCaster and needs
 *               to complete or implement the following part
 *               1) Instanciate the TCaster in the constructor from the right type
 *               2) delete the TCaster in the destructor casting it into the right type
 *               3) implement the covert method with the right caster (see any XToYCaster
 *                  as template)
 * The above is for template classes only
 * Note on Casters:
 * Caster class performs casting between same type (real to real like float to double or
 * integer to integer like int to short)
 * CasterRound class performs casting between real to integer rounding the real
 * before casting. One could use the Caster but no rounding is applied
 * CasterComplexRound same as caster round but for complex numbers
 * CasterComplexInt is used to cast complex int to complex real. (simply casting of the
 * complex numbers from int to real does not work, but from real to int works, but no rounding)
 */


class DataCaster
{
    public:
        DataCaster(){}
        virtual ~DataCaster(){}
        virtual void convert(char * in,char * out, int numEl) = 0;
        int getDataSizeIn(){return DataSizeIn;}
        int getDataSizeOut(){return DataSizeOut;}
        void * getCaster(){return TCaster;}
    protected:
        int DataSizeIn;
        int DataSizeOut;
        void * TCaster;
};

#endif //DataCaster_h
