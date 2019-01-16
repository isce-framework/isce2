#ifndef Filter_h
#define Filter_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif

#include "DataAccessor.h"
#include <stdint.h>
#include <limits>
using namespace std;

class Filter
{
    public:
        Filter()
        {
            StartLine = 0;
            EndLine = numeric_limits<int>::max();
        }
        virtual ~Filter(){}
        virtual void extract() = 0;
        void selectBand(int band){Band = band;}//used by BandExtractor
        void selectComponent(int comp){Component = comp;}//used by ComponentExtractor
        void setStartLine(int line){StartLine = line;}//set a default where nothing is done
        void setEndLine(int line){EndLine = line;}//set a default where nothing is done
        void finalize(){return;}//set a default where nothing is done
        void init(DataAccessor * in, DataAccessor * out);
    protected:
        DataAccessor * ImageIn;
        DataAccessor * ImageOut;
        int Band;
        int StartLine;
        int EndLine;
        int Component;

};

#endif //Filter_h
