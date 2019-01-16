  #ifndef DataAccessor_h
#define DataAccessor_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif

#include "InterleavedBase.h"
#include "DataCaster.h"
#include <stdint.h>
#include <iostream>
using namespace std;

class DataAccessor
{
    public:
        DataAccessor(){}           
        virtual ~DataAccessor(){}
        virtual double getPx2d(int row, int col) = 0;
        virtual double getPx1d(int pos) = 0;
        virtual int getLine(char * buf, int  pos) = 0;
        virtual int getLineBand(char * buf, int pos, int band) = 0;
        virtual void setLine(char * buf, int pos) = 0;
        virtual void setLineBand(char * buf, int pos, int band) = 0;
        virtual void setLineSequential(char * buf) = 0;
        virtual void setLineSequentialBand(char *buf, int band) = 0;
        virtual void setStream(char * dataLine,  int & numEl) = 0;
        virtual void setStreamAtPos(char * dataLine, int &  pos,  int & numEl) = 0;
        virtual void setSequentialElements(char * buf, int row, int col, int numEl) = 0;
        virtual void getStream(char * dataLine,  int  & numEl) = 0;
        virtual void getStreamAtPos(char * dataLine, int & pos,  int & numEl) = 0;
        virtual void getSequentialElements(char * buf, int row, int col, int & numEl) = 0;
        virtual int getLineSequential(char * buf) = 0;
        virtual int getLineSequentialBand(char* buf, int band) = 0;
        virtual void finalize() = 0;
        void rewindAccessor();
        void alloc(int numLines);
        void createFile(int numLines);
        void initSequentialAccessor(int line);
        int getWidth(){return LineWidth;}
        int getBands(){return Bands;}
        int getSizeIn(){return DataSizeIn;}
        int getSizeOut(){return DataSizeOut;}
        int getNumberOfLines(){return NumberOfLines;}
        int getLineOffset(){return LineOffset;}
        string getFilename(){return Accessor->getFilename();}
        void setLineOffset(int lineoff){LineOffset = lineoff;}

        InterleavedBase * getInterleavedAccessor(){return Accessor;}
        DataCaster * getDataCaster(){return Caster;}
    protected:
        InterleavedBase * Accessor;
        DataCaster * Caster;
        
        /**
         * Size of the destination data.
         **/
        int DataSizeOut;
        /**
         * Size of the source data.
         **/
        int DataSizeIn;
        
        /**
         * Number of bands for the adopted interleaved scheme.
         **/
        int Bands;
        /**
         * Number of pixels per line.
         **/
        int LineWidth;
        /**
         * LineSequential Counter.
         **/
        int LineCounter;

        /**
         * Polynomial Interpolator type
         */
        void * poly;
        /**
         * Number of lines
         */
        int NumberOfLines;
        /**
         * Initial element when accessing a line
         */
        int LineOffset;
};

#endif //DataAccessor_h
