#ifndef InterleavedBase_h
#define InterleavedBase_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif

#include <stdlib.h>
#include <string>
#include <stdexcept>


using namespace std;

class InterleavedBase
{
    public:
        InterleavedBase(){
            EofFlag = 0;
            Data = NULL;
            NumberOfLines = 0;
                            }
        virtual ~InterleavedBase(){}
        /**
         * Get the numEl pixels  from the Fin stream starting from the position (row,col). The number of rows and columns are zero based.
         **/
        virtual void getData(char * buf,int row, int col, int & numEl) = 0;
        virtual void getDataBand(char *buf,int row, int col, int &numEl, int band) = 0;
        virtual void setData(char * buf,int row, int col, int  numEl) = 0;
        virtual void setDataBand(char * buf, int row, int col, int numEl, int band) = 0;
        virtual void init(void * poly) = 0;
        virtual void init(string filename,string accessMode,int sizeV,int Bands, int LineWidth) = 0;

        virtual void getStreamAtPos(char * buf,int & pos,int & numEl) = 0;
        virtual void setStreamAtPos(char * buf,int & pos,int & numEl) = 0;
        virtual void getStream(char * buf,int & numEl) = 0;
        virtual void setStream(char * buf,int numEl) = 0;
        virtual void rewindAccessor() = 0;
        virtual void createFile(int numberOfLine) = 0;
        virtual int getFileLength() = 0;

        virtual void finalize() = 0;
        void alloc(int numLines);

        void setLineWidth(int lw){LineWidth = lw;}
        void setDataSize(int ds){SizeV = ds;}
        void setBands(int bd){Bands = bd;}
        void setNumberOfLines(int nl){NumberOfLines = nl;}
        int getLineWidth(){return LineWidth;}
        int getDataSize(){return SizeV;}
        int getBands(){return Bands;}
        int getEofFlag(){return EofFlag;}
        int getNumberOfLines(){return NumberOfLines;}
        string getFilename() {return Filename;}

        void setAccessMode(string accessMode);
        string getAccessMode(){return AccessMode;};


    protected:
        /**
         * Name associated with the image file.
         *
         **/
        string Filename;
        /**
         * Size of the DataType or ID of the datatype.
         **/
        int SizeV;

        /**
         * Number of bands for the adopted interleaved scheme.
         **/
        int Bands;

        /**
         * Number of pixels per line.
         **/
        int LineWidth;

        /**
         * Number of  lines.
         **/
        int NumberOfLines;

        /**
         * Access mode of the underlaying file object.
         **/
        string AccessMode;
        /**
         * Flag that is set to 1 when the EOF is reached. 
         **/

        int EofFlag;
        /**
         * Flag that is set to 1 when the good() stream method returns false. 
         **/

        int NoGoodFlag;

        char * Data;

};

#endif //InterleavedAccessor_h
