#ifndef LineAccessorF_h
#define LineAccessorF_h


#include "LineAccessorFFortTrans.h"
#include "LineAccessor.h"
#include <string>
#include <stdint.h>

using namespace std;
/**
  * @file
  * This is a C interface that allows fortran code to call public methods of a LineAccessor object.

  * The functions name in fortran will be the same except for the suffix "_f" that needs to be removed.
  * Moreover each function "func(args)" will be invoked from fortran using the syntax: call func(args).
  * The correspondence between C and fortran data types is:
  *     - uint64_t * <--> integer*8.
  *     - char * <--> character*X (X integer number).
  *     - int * <--> integer or integer*4.
  * @see LineAccessor.cpp
**/
extern "C"
{
        /**
          * Creates a LineAccessor object. The address of the object is stored in (*ptLineAccessor) and returned to fortran. Each
          subsequent call in fortran to access methods of this object needs to pass this value as first argument.
          * @param  ptLineAccessor: the value (*ptLineAccessor) is the address of the LineAccessor object just created.
        **/
          void getLineAccessorObject_f(uint64_t * ptLineAccessor);

        /**
          * Returns the endianness of the machine running the code. Does not require that initLineAccessor() be called
          * before execution.
          * @param  ptLineAccessor the value (*ptLineAccessor) is the address of the LineAccessor object.
          * @param  endian it is set to 'b' for big endian or 'l' for little endian.
        **/

          void  getMachineEndianness_f(uint64_t * ptLineAccessor, char * endian);

        /**
          * Initializes LineAccessor object.
          * @param  ptLineAccessor the value (*ptLineAccessor) is the address of the LineAccessor object.
          * @param filename name of the file to be accessed.
          * @param  filemode access mode of the file.
          * @param  endianFile endiannes of the data stored in the file. Values are 'b' or 'B' for big endian and 'l' or 'L' for little endian.
          * @param  type file data type.
          * @param  row pointer to number of rows of the buffer tile. Set it to one if no tiling is desired.
          * @param  col pointer to number of columns of the buffer tile. It must be equal to the number of columns of the associated file.
          * @see  printAvailableDataTypesAndSizes_f().
          * @see  LineAccessor::AccessMode.
          * @see  LineAccessor::FileDataType.
        **/
        void initLineAccessor_f(uint64_t * ptLineAccessor, char * filename, char * filemode, char * endianFile, char * type, int * row, int * col, long int filenameLenght, long int filemodeLength, long int pass, long int typeLength);

        /**
          *  Changes the file format from BandSchemeIn to BandSchemeOut. Possible formats are BSQ = 1 BIP = 2 and BIL = 3. Does not require that initLineAccessor() be called
          * before execution.
          * \note When calling the function from fortran only the parameters listed in the \b Parameters section need to be passed.
          * The remaining arguments are hidden parameters that correspond to the lengths of the char * passed.

          * @param  ptLineAccessor the value (*ptLineAccessor) is the address of the LineAccessor object.
          * @param  filein input filename.
          * @param  fileout output filename.
          * @param type variable type (FLOAT, INT etc).
          * @param width pointer to number of columns in the file.
          * @param  numBands pointer to number of bands for the interleaved schemes.
          * @param  bandIn the value (*bandIn) is the input interleaved scheme.
          * @param  bandOut the value (*bandOut) is the output interleaved scheme.
          * @see  printAvailableDataTypesAndSizes_f().
          * @see initLineAccessor_f().

        **/

void changeBandScheme_f(uint64_t * ptLineAccessor, char * filein, char * fileout, char * type, int * width, int * numBands, int * bandIn, int * bandOut, long int fileinLength, long int fileoutLength, long int typeLength);

        /**
          * Changes the file endiannes. Does not require that initLineAccessor() be called
          * before execution.
          * \note When calling the function from fortran only the parameters listed in the \b Parameters section need to be passed.
          * The remaining arguments are hidden parameters that correspond to the lengths of the char * passed.

          * @param  ptLineAccessor the value (*ptLineAccessor) is the address of the LineAccessor object.
          * @param  filein input filename.
          * @param  fileout output filename.
          * @param  type variable type (FLOAT, INT etc).
          * @see  printAvailableDataTypesAndSizes_f().
        **/
        void convertFileEndianness_f(uint64_t * ptLineAccessor,char * filein, char * fileout, char * type, long int fileinLength, long int fileoutLength, long int typeLength);

        /**
          * Always call this function if a LineAccessor object was created. It deletes the pointer to the object, closes the file associated with the object, frees memory and
          * possibly flushes unfilled buffer tiles to disk.
          * @param ptLineAccessor the value (*ptLineAccessor) is the address of the LineAccessor object.
          * @see  getLineAccessorObject_f().
        **/
        void finalizeLineAccessor_f(uint64_t * ptLineAccessor);

        /**
          * For a file object opened in write or writeread mode it creates a blank file of size LineSize*(*lenght).
          * @param  ptLineAccessor the value (*ptLineAccessor) is the address of the LineAccessor object.
          * @param length the value (*length) is the number of lines in the file.
          * @see LineAccessor::LineSize.
        **/
        void createFile_f(uint64_t * ptLineAccessor,int * length);

    /**
     * Reset some class variable so that the image can be reused. If one wants to use the same image wit different access mode, then create a new object with the new access mode.
     *
    **/
    void rewindImage_f(uint64_t * ptLineAccessor);

        /**
          * Set the initial line to use getLineSequential_f().
          * @param  ptLineAccessor the value (*ptLineAccessor) is the address of the LineAccessor object.
          * @param begLine the value (*begLine) is the initial line. Default is one.
          * @see  getLineSequential_f().
        **/
        void initSequentialAccessor_f(uint64_t * ptLineAccessor, int * begLine);

        /**
          * Prints the available data types and their sizes.
          * Does not require that initLineAccessor_f() be called.
          * @param  ptLineAccessor the value (*ptLineAccessor) is the address of the LineAccessor object.
        **/
        void printAvailableDataTypesAndSizes_f(uint64_t *  ptLineAccessor);

        /**
          * Provides the size of the file datatype.
          * \note When calling the function from fortran only the parameters listed in the \b Parameters section need to be passed.
          * The remaining arguments are hidden parameters that correspond to the lengths of the char * passed.

          * @param  ptLineAccessor the value (*ptLineAccessor) is the address of the LineAccessor object.
          * @param type data type.
          * @param size the value (*size) contains the size of the data type.

        **/
        void getTypeSize_f(uint64_t *  ptLineAccessor, char * type, int * size, long int len);
        /**
          * Provides the number of columns of the file associated with the accessor object.
          * @param  ptLineAccessor the value (*ptLineAccessor) is the address of the LineAccessor object.
          * @param lineWidth the value (*lineWidth) contains the file width.

        **/
        void getFileWidth_f(uint64_t *  ptLineAccessor, int * lineWidth);

        /**
          * Provides the number of lines of the file associated with the accessor object.
          * @param  ptLineAccessor the value (*ptLineAccessor) is the address of the LineAccessor object.
          * @param  length the value (*length) contains the file lenght.

        **/
        void getFileLength_f(uint64_t *  ptLineAccessor, int * length);

        /**
          * Prints a series of information related to the file associated with the accessor.
          * @param  ptLineAccessor the value (*ptLineAccessor) is the address of the LineAccessor object.
        **/
        void printObjectInfo_f(uint64_t *  ptLineAccessor);

        /** For each call it sets a line from the dataLine character array to the associated file object starting from a given line. The starting line is
          * set using initSequentialAccessor(). The default starting line is one.
          * @param ptLineAccessor the value (*ptLineAccessor) is the address of the LineAccessor object.
          * @param dataLine character array containing the data to be set.
          * @see  getLineSequential_f().
          * @see  initSequentialAccessor_f().
        **/
        void setLineSequential_f(uint64_t * ptLineAccessor,  char * dataLine);

        /**
          * Gets  the line at position (*row)  from the associated file object and puts it in the
          * character array dataLine.
          * @param ptLineAccessor the value (*ptLineAccessor) is the address of the LineAccessor object.
          * @param  dataLine character array where read data are put.
          * @param row the value (*row) is the line number in the file. If the line is out of bounds then (*row) = -1.
        **/
        void getLine_f(uint64_t * ptLineAccessor,char * dataLine, int * row);
        /**
          * Sets  (*numEl) elements from the associated file object. The first access is at the beginning of the file. All the subsequent accesses are
          * at the next element of the last one previously accessed.
          * @param  dataLine character array where read data are put.
          * @param  numEl at the function call the value (*numEl) is the number of elements to be read.
          * @see  setSteamAtPos_f().
          * @see  getSteamAtPos_f().
          * @see  getSteam_f().
        **/

        void setStream_f(uint64_t * ptLineAccessor,  char * dataLine, int * numEl);

        /**
          * Sets  (*numEl) elements from the associated file object at position (*pos). The position is in unit of the FileDataType and NOT in bytes.
          * @param  dataLine character array where read data are put.
          * @param  numEl at the function call the value (*numEl) is the number of elements to be read.
          * @see  setSteamAtPos_f().
          * @see  getSteamAtPos_f().
          * @see  getSteam_f().
          * @see  FileDataType.
        **/
        void setStreamAtPos_f(uint64_t * ptLineAccessor,  char * dataLine, int * pos, int * numEl);

        /**
          * Gets  (*numEl) elements from the associated file object. The first access is at the beginning of the file. All the subsequent accesses are
          * at the next element of the last one previously accessed.
          * @param  dataLine character array where read data are put.
          * @param  numEl at the function call the value (*numEl) is the number of elements to be read. At the return from the function call it's
          * the number of elements actually read. Check if (*numEl) before and after the function call differs to know when the end of file is reached.
          * @see  getSteamAtPos_f().
          * @see  setSteamAtPos_f().
          * @see  setSteam_f().
        **/

        void getStream_f(uint64_t * ptLineAccessor,  char * dataLine, int * numEl);
        /**
          * Gets  (*numEl) elements from the associated file object at position (*pos). The position is in unit of the FileDataType and NOT in bytes.
          * @param  dataLine character array where read data are put.
          * @param  numEl at the function call the value (*numEl) is the number of elements to be read. At the return from the function call it's
          * the number of elements actually read. Check if (*numEl) before and after the function call differs to know when the end of file is reached.
          * @see  getSteamAtPos_f().
          * @see  setSteamAtPos_f().
          * @see  setSteam_F().
        **/

        void getStreamAtPos_f(uint64_t * ptLineAccessor,  char * dataLine, int * pos, int * numEl);

        /**
          * Sets  (*numEl) elements from the character array dataLine to the associated file object starting from the position column = (*col) and row = (*row).
          * If the full file is not accessed sequentially (i.e. random access), make sure that the file is already created using createFile_f() and that the access mode is "readwrite".
          * @param  ptLineAccessor the value (*ptLineAccessor) is the address of the LineAccessor object.
          * @param  dataLine character array where the data are.
          * @param  row the value (*row) is the row position in the file.
          * @param col the value (*col) is the column position int the file.
          * @param numEl the value (*numEl) is the number of elements to be set.
          * @see  getSequentialElements_f().
        **/

        void setSequentialElements_f(uint64_t * ptLineAccessor,  char * dataLine, int * row, int * col, int * numEl);

        /**
          * Puts (*numEl) elements in dataLine in the associated file object at the positions column = col[i] and row = row[i]  (for the i-th element).
          * Make sure that the file is already created using createFile_f() and that the access mode is "readwrite".
          * @param  ptLineAccessor the value (*ptLineAccessor) is the address of the LineAccessor object.
          * @param dataLine character array containing the data.
          * @param row array with the row positions of the elements to be set.
          * @param  col array with the column positions of the elements to be set.
          * @param numEl the value (*numEl) is the number of elements to be set.
          * @see  getElements_f().
          * @see  createFile_f().
          * @see  LineAccessor::openFile().
        **/
        void setElements_f(uint64_t * ptLineAccessor,  char * dataLine, int * row, int * col, int * numEl);

        /**
          * Sets  a line at the position  (*row).
          * If the full file is not accessed sequentially (i.e. random access), make sure that the file is already created using createFile() and that the access mode is "readwrite".
          * @param  ptLineAccessor the value (*ptLineAccessor) is the address of the LineAccessor object.
          * @param  dataLine character array where the data are.
          * @param  row the value (*row) is the line number in the file.
        **/
        void setLine_f(uint64_t * ptLineAccessor,char * dataLine, int * row);
        /**
          * For each call it gets a line from the associated file object and puts it in the character array dataLine starting from a given line. The starting
          * line is set using initSequentialAccessor(). The default starting line is one.
          * @param  ptLineAccessor the value (*ptLineAccessor) is the address of the LineAccessor object.
          * @param dataLine character array where read data are put.
          * @param  eof the value (*eof) is set to -1 when the end of file is reached otherwise it give the position of the line just read.
          * @see  setLineSequential_f().
          * @see  initSequentialAccessor_f().
        **/
        void getLineSequential_f(uint64_t * ptLineAccessor,  char * dataLine, int * eof);

        /**
          * Gets  (*numEl) elements from the associated file object starting from the position column = (*col) and row = (*row) and puts them in the
          * character array dataLine. Note the (*numEl) and (*col) refer to the particular FileDataType. Reading (*numEl) elements correspond to reading
          * (*numEl)*(sizeof(FileDataType)) bytes. An element at colomn (*col) starts at the byte position (*col)*(sizeof(FileDataType)) of a given row.
          * @param ptLineAccessor the value (*ptLineAccessor) is the address of the LineAccessor object.
          * @param  dataLine character array where read data are put.
          * @param row the value (*row) is the row position.
          * @param  col the value (*col) is the column position.
          * @param  numEl at the function call the value (*numEl) is the number of elements to be read. At the return from the function call it's
          * the number of elements actually read. Check if (*numEl) before and after the function call differs to know when the end of file is reached.
          * @see  setSequentialElements_f().
          * @see  LineAccessor::FileDataType.
        **/
        void getSequentialElements_f(uint64_t * ptLineAccessor,  char * dataLine, int * row, int * col, int * numEl);

        /**
          * Gets  (*numEl) elements from the associated file object whose positions are at column = col[i] and row = row[i]  (for the i-th element)
          * and puts it in the character array dataLine. Note the (*numEl) and (*col) refer to the particular FileDataType. Reading (*numEl)
          * elements corresponds to reading (*numEl)*(sizeof(FileDataType)) bytes. An element at colomn col[i]  starts at the byte position
          * col[i]*(sizeof(FileDataType)) of a given row.
          * @param ptLineAccessor the value (*ptLineAccessor) is the address of the LineAccessor object.
          * @param  dataLine character array where read data are put.
          * @param row array with the row positions of the elements to be read.
          * @param col array with the column positions of the elements to be read.
          * @param numEl at the function call the value (*numEl) is the number of elements to be read. At the return from the function call
          * it's the number of elements actually read. Check if (*numEl) before and after the function call differs to know when the end of file is reached.
          * @see  setElements_f().
          * @see  LineAccessor::FileDataType.
        **/
        void getElements_f(uint64_t * ptLineAccessor,  char * dataLine, int * row, int * col, int * numEl);

}
        /**
            Since the char * is passed from fortran where the array is not NULL terminated, it reads it up to the first blank encountered and put it into a C string.
            \note This function is not meant to be called from fortran.

            @param word character array.
            @param len length of the character array as decleared in the fortran code.
            @return \c string the array word with trailing blank removed.

        **/
        string getString(char * word, long int len);
        /**
            Converts an integer type to the corresponding BandSchemeType.
            \note This function is not meant to be called from fortran.

            @param band band interleaved scheme of integer type.
            @return \c BandSchemeType band interleaved scheme of enum type.
            @see changeBandScheme_f().
            @see BandSchemeType.
        **/
        BandSchemeType convertIntToBandSchemeType(int band);


#endif //LineAccessorF_h
