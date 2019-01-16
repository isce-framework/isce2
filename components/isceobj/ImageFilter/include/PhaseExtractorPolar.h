#ifndef PhaseExtractorPolar_h
#define PhaseExtractorPolar_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif
#include "Filter.h"
#include "DataAccessor.h"
#include <stdint.h>
using namespace std;

class PhaseExtractorPolar : public Filter
{
    public:
        PhaseExtractorPolar(){}
        ~PhaseExtractorPolar(){}
        void extract();
    protected:

};

#endif //PhaseExtractorPolar_h
