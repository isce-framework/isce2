#ifndef DataAccessorNoCaster_h
#define DataAccessorNoCaster_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif

#include "InterleavedBase.h"
#include "DataAccessor.h"
#include "DataCaster.h"
#include <stdint.h>
using namespace std;

class DataAccessorNoCaster: public DataAccessor
{
    public:
        DataAccessorNoCaster(InterleavedBase * accessor)
        {
            Accessor = accessor;
            LineWidth = Accessor->getLineWidth();
            Bands = Accessor->getBands();
            DataSizeIn  = Accessor->getDataSize();
            DataSizeOut  = DataSizeIn;
            LineCounter = 0;
            LineOffset = 0;
            NumberOfLines = Accessor->getNumberOfLines();
        }           
        ~DataAccessorNoCaster(){}
        void getStreamAtPos(char * buf,int & pos,int & numEl);
        void setStreamAtPos(char * buf,int & pos,int & numEl);
        void getStream(char * buf,int & numEl);
        void setStream(char * buf,int & numEl);
        int getLine(char * buf, int pos);
        int getLineBand(char * buf, int pos, int band);
        void setLine(char * buf, int pos);
        void setLineBand(char * buf, int pos, int band);
        void setLineSequential(char * buf);
        void setLineSequentialBand(char * buf, int band);
        int getLineSequential(char * buf);
        int getLineSequentialBand(char * buf, int band);
        void getSequentialElements(char * buf, int row, int col, int & numEl);
        void setSequentialElements(char * buf, int row, int col, int numEl);
        void finalize();
        double getPx2d(int row, int col);
        double getPx1d(int pos);

    protected:
        
};

#endif //DataAccessorNoCaster_h
