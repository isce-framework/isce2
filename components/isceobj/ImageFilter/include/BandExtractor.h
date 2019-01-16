#ifndef BandExtractor_h
#define BandExtractor_h

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

class BandExtractor : public Filter
{
    public:
        BandExtractor(){}
        ~BandExtractor(){}
        void extract();
        void selectBand(int band){Band = band;}
    protected:

};

#endif //BandExtractor_h
