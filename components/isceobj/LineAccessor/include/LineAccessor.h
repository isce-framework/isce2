#ifndef LineAccessor_h
#define LineAccessor_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif



#include <cmath>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <complex>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

using namespace std;

/**
  *  Enum type to characterize the different interleaved schemes of images.
  *  See parameters descriptions.
*/

enum BandSchemeType
{
    BNULL = 0,/**< Interleaving scheme undefined.*/
    BSQ = 1,  /**< Band Sequential or Interleaved.*/
    BIP = 2,  /**< Band Interleaved by Pixel.*/
    BIL = 3 /**< Band Interleaved by Line. */


};

/** \brief
  * Class to handle read and write into file.

  * This class provides methods to read or write data (sequentially or randomly) from or to a file. Some optimizations are implemented such as buffering the
  * data to be read or written. It also provides methods to change the interleaving scheme adopted to store the data into the file and change their endianness. Note that row and column numbers are one based and not zero based.
  * See the public methods for more details.
**/
// class begin
class LineAccessor
{
    public:
        /// Constructor.
        LineAccessor():ColumnPosition(1),FileLength(0),IsInit(false),LineCounter(1),LinePosition(1),MachineSize(32),NeedToFlush(false),ReadBufferSize(1)
        {
        }
        /// Destructor.
        ~LineAccessor()
        {
        }
        /**
          *  Changes the file format from BandSchemeIn to BandSchemeOut. Possible formats are BSQ BIL and BIP. Does not require that initLineAccessor() be called
          * before execution.
          * @param filein input filename.
          * @param fileout output filename.
          * @param type variable type (FLOAT, INT etc).
          * @param width number of columns in the file.
          * @param numBands number of bands for the interleaved schemes.
          * @param  bandIn input interleaved scheme.
          * @param bandOut output interleaved scheme.
          * @see BandSchemeType.
          * @see getAvailableDataTypes().
          * @see getAvailableDataTypesAndSizes().
          * @see initLineAccessor().

        **/
        void changeBandScheme(string filein, string fileout, string type, int width, int numBands, BandSchemeType bandIn, BandSchemeType bandOut);

        /**
          *  Changes the file endiannes.
          * @param  filein input filename.
          * @param  fileout output filename.
          * @param  type variable type (FLOAT, INT etc).
          * @see  getAvailableDataTypes().
          * @see  getAvailableDataTypesAndSizes().
        **/
        void convertFileEndianness(string filein, string fileout, string type);

        /**
          * For a file object opened in write or writeread mode it creates a blank file of size LineSize*(*fileLenght).
          * @param fileLength the value (*fileLength) is the number of lines in the file.
          * @see LineSize.
        **/
        void createFile(int * fileLength);


    /**
     * Reset some class variable so that the image can be reused. If one wants to use the same image wit different access mode, then create a new object with the new access mode.
     *
    **/

    void rewindImage();

        /**
          * Returns the endianness of the machine running the code. Does not require that initLineAccessor() be called
          * before execution.
          * @return \c char 'b' for big endian and 'l' for little endian.
        **/

        char getMachineEndianness();

        /**
          * Returns the character array associated with the buffer tile.
          * @return \c char * pointer to buffer tile.
        **/
        char * getTileArray();
        /**
          * Returns the size of the data type "type".
          * Does not require that initLineAccessor() be called.
          * @param type data type.
          * @return \c int size of type.
          * @see  getSizeForSwap().
          * @see  getAvailableDataTypes().
          * @see  getAvailableDataTypesAndSizes().
        **/

        int getTypeSize(string type);
        /**
          * Returns the size of the data type "type" used for byte swapping. For built in data types the returned value is the same as the one from  getTypeSize().
          * Does not require that initLineAccessor() be called.
          * For complex types the returned value is half.
          * @param type data type.
          * @return \c int size of type for byte swapping.
          * @see  getTypeSize().
          * @see  getAvailableDataTypes().
          * @see  getAvailableDataTypesAndSizes().
        **/
        int getSizeForSwap(string type);

        /**
          * Returns a vector of strings with all the data types supported.
          * Does not require that initLineAccessor() be called.
          * @return \c vector<string> vector of strings containing data types supported.
          * @see  getAvailableDataTypesAndSizes().
        **/

        vector<string> getAvailableDataTypes();

        /**
          * Provides a vector of strings with all the data types supported and a vector of integers with the corresponding sizes.
          * Does not require that initLineAccessor() be called.
          * @param types reference to the vector of strings where the data types are put.
          * @param  size reference to the vector of integers where the sizes of the corresponding data types  are put.
          * @see  getAvailableDataTypes().
        **/
        void getAvailableDataTypesAndSizes(vector<string> & types, vector<int> & size);

        /**
          * Always call this function if initLineAccessor() was called at the beginning. It closes the file associated with the object, frees memory and
          * possibly flushes unfilled buffer tiles to disk.
          * @see  initLineAccessor().
        **/
        void finalizeLineAccessor();
        /**
          * Gets  (*numEl) elements from the associated file object. The first access is at the beginning of the file. All the subsequent accesses are
          * at the next element of the last one previously accessed.
          * @param  dataLine character array where read data are put.
          * @param  numEl at the function call the value (*numEl) is the number of elements to be read. At the return from the function call it's
          * the number of elements actually read. Check if (*numEl) before and after the function call differs to know when the end of file is reached.
          * @see  getSteamAtPos().
          * @see  setSteamAtPos().
          * @see  setSteam().
        **/
        void getStream(char * dataLine,  int * numEl);

        /**
          * Gets  (*numEl) elements from the associated file object at position (*pos). The position is in unit of the FileDataType and NOT in bytes.
          * @param  dataLine character array where read data are put.
          * @param  numEl at the function call the value (*numEl) is the number of elements to be read. At the return from the function call it's
          * the number of elements actually read. Check if (*numEl) before and after the function call differs to know when the end of file is reached.
          * @see  getSteamAtPos().
          * @see  setSteamAtPos().
          * @see  setSteam().
          * @see  FileDataType.
        **/
        void getStreamAtPos(char * dataLine, int * pos,  int * numEl);

        /**
          * Gets  (*numEl) elements from the associated file object whose positions are at column = col[i] and row = row[i]  (for the i-th element)
          * and puts it in the character array dataLine. Note the (*numEl) and (*col) refer to the particular FileDataType. Reading (*numEl)
          * elements corresponds to reading (*numEl)*(sizeof(FileDataType)) bytes. An element at colomn col[i]  starts at the byte position
          * col[i]*(sizeof(FileDataType)) of a given row. Note: this method is slow and sometime is better to access the elements sequentially with getSequntialElemetns() or by line with getLineSequential() or getLine() and then pick the desired elemnts.
          * @param  dataLine character array where read data are put.
          * @param  row array with the row positions of the elements to be read.
          * @param  col array with the column positions of the elements to be read.
          * @param numEl at the function call the value (*numEl) is the number of elements to be read.
          * @see  setElements().
          * @see  FileDataType.
        **/


        void getElements(char * dataLine, int * row, int * col, int * numEl);
        /**
          * For each call it gets a line from the associated file object and puts it in the character array dataLine starting from a given line. The starting
          * line is set using initSequentialAccessor(). The default starting line is one.
          * @param  dataLine character array where read data are put.
          * @param  eof the value (*eof) is set to -1 when the end of file is reached otherwise it give the position of the line just read.
          * @see  setLineSequential().
          * @see  initSequentialAccessor().
        **/
        void getLineSequential(char * dataLine, int * eof);

        /**
          * Provides the number of lines of the file associated with the accessor object.
          * @param  length the value (*length) contains the file lenght.

        **/

        inline void getFileLength(int * length)
        {
                (*length) =     FileLength;
        }
        /**
          * Provides the number of columns of the associated file.
          * @param  width the value (*width) contains the file width.

        **/
        inline void getFileWidth(int * width)
        {
                (*width) =      FileWidth;
        }

        /**
          * Returns the machine architecture size (32 or 64).
          * @return \c int architecture size.

        **/
        inline int getMachineSize()
        {
            return (sizeof(long long int) == 8 ? 64 : 32);
        }


        /**
          * Gets  (*numEl) elements from the associated file object starting from the position column = (*col) and row = (*row) and puts them in the
          * character array dataLine. Note the (*numEl) and (*col) refer to the particular FileDataType. Reading (*numEl) elements correspond to reading
          * (*numEl)*(sizeof(FileDataType)) bytes. An element at colomn (*col) starts at the byte position (*col)*(sizeof(FileDataType)) of a given row.
          * @param  dataLine character array where read data are put.
          * @param row the value (*row) is the row position.
          * @param  col the value (*col) is the column position.
          * @param  numEl at the function call the value (*numEl) is the number of elements to be read. At the return from the function call it's
          * the number of elements actually read. Check if (*numEl) before and after the function call differs to know when the end of file is reached.
          * @see  FileDataType.
        **/
        void getSequentialElements(char * dataLine, int * row, int * col, int * numEl);

        /**
          * Gets  the line at position (*row)  from the associated file object and puts it in the
          * character array dataLine.
          * @param  dataLine character array where read data are put.
          * @param row the value (*row) is the line number in the file. If the line is out of bounds then (*row) = -1.
        **/
        void getLine(char * dataLine, int * row);

        /**
          * Checks if the initLineAccessor() method has been invoked before.
          * @return \c bool true if the initLineAccessor() method has been invoked before, false otherwise.
        **/

        bool isInit();

        /**
          * Initializes the accessor object. If the col parameter (i.e. the width) is unknown and the file is randomly accessed through the set,getStream's function, set it to any integer number.
          * @param  filename name of the file to be accessed.
          * @param  filemode access mode of the file.
          * @param endianFile endiannes of the data stored in the file. Values are 'b' or 'B' for big endian and 'l' or 'L' for little endian.
          * @param  type file data type.
          * @param row number of rows of the buffer tile. Set it to one if no tiling is desired.
          * @param col number of columns of the buffer tile. It must be equal to the number of columns of the associated file.
          * @see  getAvailableDataTypes().
          * @see  getAvailableDataTypesAndSizes().
          * @see  AccessMode.
          * @see  FileDataType.
          * @see  getSteamAtPos().
          * @see  getSteam().
          * @see  setSteamAtPos().
          * @see  setSteam().
        **/
        void initLineAccessor(string filename, string filemode, char endianFile, string type, int row, int col);

        /**
          * Set the initial line to use getLineSequential().
          * @param  begLine the value (*begLine) is the initial line. Default is one.
          * @see  getLineSequential().
        **/
        void initSequentialAccessor(int * begLine);

        /**
          * Prints the available data types and their sizes.
          * @see  getAvailableDataTypes().
          * @see  getAvailableDataTypesAndSizes().
        **/
        void printAvailableDataTypesAndSizes();
        /**
          * Prints a series of information related to the file associated with the accessor.
        **/
        void printObjectInfo();


        /**
          * Sets  (*numEl) elements from the associated file object. The first access is at the beginning of the file. All the subsequent accesses are
          * at the next element of the last one previously accessed.
          * @param  dataLine character array where read data are put.
          * @param  numEl at the function call the value (*numEl) is the number of elements to be read.
          * @see  setSteamAtPos().
          * @see  getSteamAtPos().
          * @see  getSteam().
        **/
        void setStream(char * dataLine,  int * numEl);

        /**
          * Sets  (*numEl) elements from the associated file object at position (*pos). The position is in unit of the FileDataType and NOT in bytes.
          * @param  dataLine character array where read data are put.
          * @param  numEl at the function call the value (*numEl) is the number of elements to be read.
          * @see  setSteamAtPos().
          * @see  getSteamAtPos().
          * @see  getSteam().
          * @see  FileDataType.
        **/
        void setStreamAtPos(char * dataLine, int * pos,  int * numEl);


        /**
          * Puts (*numEl) elements in dataLine in the associated file object at the positions column = col[i] and row = row[i]  (for the i-th element).
          * Make sure that the file is already created using createFile() and that the access mode is "readwrite".
          * @param  dataLine character array containing the data.
          * @param  row array with the row positions of the elements to be set.
          * @param col array with the column positions of the elements to be set.
          * @param numEl the value (*numEl) is the number of elements to be set.
          * @see  getElements().
          * @see  createFile().
          * @see  openFile().
        **/

        void setElements(char * dataLine, int * row, int * col, int * numEl);

        /**
          * Sets  a line at the position  (*row).
          * If the full file is not accessed sequentially (i.e. random access), make sure that the file is already created using createFile() and that the access mode is "readwrite".
          * @param  dataLine character array where the data are.
          * @param  row the value (*row) is the line number in the file.
        **/
        void setLine(char * dataLine, int * row);
        /** For each call it sets a line from the dataLine character array to the associated file object starting from a given line. The starting line is
          * set using initSequentialAccessor(). The default starting line is one.
          * @param  dataLine character array containing the data to be set.
          * @see  getLineSequential().
          * @see  initSequentialAccessor().
        **/
        void setLineSequential(char * dataLine);

        /**
          * Sets  (*numEl) elements from the character array dataLine to the associated file object starting from the position column = (*col) and row = (*row).
          * If the full file is not accessed sequentially (i.e. random access), make sure that the file is already created using createFile() and that the access mode is "readwrite".
          * @param  dataLine character array where the data are.
          * @param  row the value (*row) is the row position in the file.
          * @param col the value (*col) is the column position int the file.
          * @param numEl the value (*numEl) is the number of elements to be set.
        **/
        void setSequentialElements(char * dataLine, int * row, int * col, int * numEl);


        protected:

        //begin initialization list attributes
        //The following 8 attributes are in the constructor initialization list.
        //They should be declared in the same order as in the initialization list
        //to prevent the compiler from throwing a warning.

        /**
          * Keeps track of the column position where the next write from setSequentialElement() starts.
          @see setSequentialElements().
        **/
        int ColumnPosition;


        /**
          * Number of lines in the file.
          * @see getFileLength().
        **/
        int FileLength;

        /**
          * Set to true if initLineAccessor() method hab been invoked.
        **/
        bool IsInit;

        /**
          * Current line position in the tile buffer.
        **/
        int LineCounter;

        /**
          * Contains the next line position where to read for getSequentialElements() or the line where to write for setSequentialElements().
          @see getSequentialElements().
          @see setSequentialElements().
        **/
        int LinePosition;
        //contains the next line where to read for getSequntialLine. it contains the line where to write in setSequentialElements. it's  1 based and set in fortran (but used 0 based in c)

        /**
          * Machine architecture size. Possible values 32 or 64.
          @see getMachineSize().
        **/
        int MachineSize;

        /**
        * Set to true if the tile is dirty and needs to be flushed before closing the file.
        **/
        bool NeedToFlush;

        /**
          * Number of lines buffered in setElements() or getElements().
          * @see setElements().
          * @see getElements().
        **/
        int ReadBufferSize;

        //end of initialization list attributes

        //variables

        /**
          * Endianness of the file. Possible values 'b' or 'B' for big endian and 'l' or 'L' for little endian.
        **/
        char EndianFile;
        /**
          * Endianness of the machine.
          * @see getMachineEndianness().
        **/
        char EndianMachine;
        /**
          * File stream object associate with the file.
        **/

        fstream FileObject;

        /**
          * Number of columns in the file. Also equal to SizeXTile.
          * @see getFileWidth().
          * @see SizeXTile.
        **/
        int FileWidth;

        /**
          * Number of bytes per line.
        **/
        int LineSize; //size of line in byte

        /**
          * For built in data types the is the same as SizeV but for complex type is half of it.
          * @see getSizeForSwap().
        **/

        int SizeForSwap;

        /**
          * The size of the file data type.
          * @see getTypeSize().
        **/

        int SizeV;

        /**
          * Number of columns of the buffer tile. Also equal to FileWidth.
          * @see getTypeSize().
          * @see FileWidth.
        **/

        int SizeXTile;

        /**
          * Number of rows (or lines) in the buffer tile.
        **/

        int SizeYTile;//rows of tile

        /**
          * Size in bytes of the buffer tile.
        **/

        streampos TileSize;

        /**
          * Size in bytes of the assoiciated file. Also equal to LineSize*SizeYTile.
        **/

        streampos FileSize;

        /**
          * Name of the file associated to the accessor object.
        **/

        string Filename;

        /**
          * Access mode of the associated file. Possible values are "append", "read", "readwrite", "write" and "writeread" (or same words with capital letters).
          * Note that "writeread" truncates the file to zero size  if it already exists, while "readwrite" just open it for input and output with no truncation.
        **/

        string AccessMode;

        /**
          * File data type.
          * @see  getAvailableDataTypes().
          * @see  getAvailableDataTypesAndSizes().
        **/

        string FileDataType;

        /**
          * Vector containing all the data types supported.
          * @see  getAvailableDataTypes().
          * @see  getAvailableDataTypesAndSizes().
        **/

        vector<string> DataType;


        /**
          * Buffer tile array.
        **/

        char * PtArray;

        //functions

        /**
          * Checks if the value "col" is in file column range.
          *@param col value to check.
        **/
        inline void checkColumnRange(int col);

        /**
          * Checks if the value "row" is in file row range.
          *@param row value to check.
        **/
        inline void checkRowRange(int row);

        /**
          * Returns the size of the file associated to the file stream object fin.
          * @param  fin file stream.
          * @return \c streampos file size in bytes.
        **/
        streampos getFileSize(fstream & fin);
        /**
          * Returns the size of the file "filename".
          * @param  filename name of the file.
          * @return \c streampos file size in bytes.
        **/
        streampos getFileSize(string filename);

        /**
          * Opens the file "filename" with access mode "accessMode"  and associates the correspondig file stream to fd.
          * @param filename name of the file.
          * @param  accessMode file access mode.
          * @param fd reference to the opened file.
          * @see AccessMode
        **/
        void openFile(string filename, string accessMode, fstream & fd);
        /**
          *  Sets the variable AccessMode to "accessMode".
          * @param accessMode file access mode.
          * @see AccessMode
        **/
        void setAccessMode(string accessMode);
        /**
          *  Sorts array "row" in place in increasing order using quick sort algorithm. The indexing in "col" is changed accordingly to mantain the same one as in "row".
          * @param  row array to be sorted.
          * @param  col array to be reordered accornding to the new indexing in "row".
          * @param  lo index in row of the first element to be sorted.
          * @param  hi index in row of the last element to be sorted.
        **/

        void quickSort(int * row, int * col , int * indx, int lo, int hi);

        /**
          *  Swaps the bytes of numElements of size sizeV in buffer. The swapping is done in a mirrororing way. First byte with last, second byte with second o last and so on.
          * @param  buffer array containing the data.
          * @param  numElements number of elements of size sizeV to be swapped.
          * @param sizeV number of bytes to be swapped for each of the numElements.
        **/

        void swapBytes(char * buffer, int numElements, int sizeV);
        /**
          *  Swaps two bytes. The swapping is done in registers.
          * @param x the value (*x) is the two bytes integer to be swapped.
          * @return \c uint16_t the two swapped bytes.
        **/


        inline uint16_t swap2Bytes(uint16_t * x);
        /**
          *  Swaps four bytes. The swapping is done in registers.
          * @param  x the value (*x) is the four bytes integer to be swapped.
          * @return \c uint32_t the four swapped bytes.
        **/
        inline uint32_t swap4Bytes(uint32_t * x);
        /**
          * Swaps eight bytes. The swapping is done in registers.
          * Note: for a 32 bits machine the register cannot contain eight bytes so swap8BytesSlow() is used by default. To change the default behaviour compile
          * with the option -DMACHINE_64 when using a 64 bits machine.
          * @param  x the value (*x) is the eight bytes integer to be swapped.
          * @return \c uint64_t the eight swapped bytes.
        **/
        inline uint64_t swap8BytesFast(uint64_t * x);
        /**
          * Swaps eight bytes. The swapping is done in place.
          * @param x eight bytes charactes array containing the bytes to be swapped. Swapping done in place i.e. when the function  returns x contain
          * the new byte arrangement.
        **/
        inline void swap8BytesSlow(char * x);
        /**
          * Swaps twelve bytes. The swapping is done in place.
          * @param x twelve bytes charactes array containing the bytes to be swapped. Swapping done in place i.e. when the function  returns x contains
          * the new byte arrangement.
        **/
        inline void swap12Bytes(char * x); //for some architecture size(long double) = 12
        /**
          * Swaps sixteen bytes. The swapping is done in place.
          * @param  x sixteen bytes charactes array containing the bytes to be swapped. Swapping done in place i.e. when the function  returns x contains
          * the new byte arrangement.
        **/

        inline void swap16Bytes(char * x); //for some architecture size(long double) = 12

};


#endif
//end-of-file LineAccessor_h
