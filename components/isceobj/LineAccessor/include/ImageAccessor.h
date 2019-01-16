#ifndef ImageAccessor_h
#define ImageAccessor_h

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
#include "LineAccessor.h"

using namespace std;
/**
    *This class provides a set of convenience methods for the class LineAccessor. It removes the awkwardness of passing pointers to variables instead of the variables themselves.
**/
class ImageAccessor : public LineAccessor
{
    public:
        /// Constructor.
        ImageAccessor()
        {
        }
        /// Destructor.
        ~ImageAccessor()
        {
        }

        /**
          * For a file object opened in write or writeread mode it creates a blank file of size #LineAccessor::LineSize * fileLenght.
          * @param fileLength  the number of lines in the file.
          * @see #LineAccessor::LineSize
        **/
        void createFile(int  fileLength);




        /**
          * Always call this function if initImageAccessor() was called at the beginning. It closes the file associated with the object, frees memory and
          * possibly flushes unfilled buffer tiles to disk.
          * @see  initImageAccessor().
        **/
        void finalizeImageAccessor();
        /**
          * Gets  numEl elements from the associated file object whose positions are at column = col[i] and row = row[i]  (for the i-th element)
          * and puts it in the character array dataLine. Note the numEl and col refer to the particular FileDataType. Reading numEl
          * elements corresponds to reading (numEl)*(sizeof(FileDataType)) bytes. An element at colomn col[i]  starts at the byte position
          * col[i]*(sizeof(FileDataType)) of a given row.
          * @param  dataLine character array where read data are put.
          * @param  row array with the row positions of the elements to be read.
          * @param  col array with the column positions of the elements to be read.
          * @param numEl at the function call the value numEl is the number of elements to be read.
          * @see  setElements().
          * @see  LineAccessor::FileDataType.
        **/


        void getElements(char * dataLine, int * row, int * col, int  numEl);
        /**
          * For each call it gets a line from the associated file object and puts it in the character array dataLine starting from a given line. The starting
          * line is set using initSequentialAccessor(). The default starting line is one.
          * @param  dataLine character array where read data are put.
          * @param  eof the value eof is set to -1 when the end of file is reached otherwise it give the position of the line just read.
          * @see  setLineSequential().
          * @see  initSequentialAccessor().
        **/
        void getLineSequential(char * dataLine, int  & eof);

        /**
          * Provides the number of lines of the file associated with the accessor object.
          * @return  \c int  file lenght.

        **/

        inline int getFileLength()
        {
                return FileLength;
        }
        /**
          * Provides the number of columns of the associated file.
          * @return \c int file width.

        **/
        inline int getFileWidth()
        {
                return FileWidth;
        }

        /**
          * Gets  numEl elements from the associated file object starting from the position column = col and row = row and puts them in the
          * character array dataLine. Note the numEl and coli refer to the particular LineAccessor::FileDataType. Reading numEl elements correspond to reading
          * numEl*(sizeof(FileDataType)) bytes. An element at colomn col starts at the byte position col*(sizeof(FileDataType)) of a given row.
          * @param  dataLine character array where read data are put.
          * @param row  the row position.
          * @param  col the column position.
          * @param  numEl at the function call the value numEl is the number of elements to be read. At the return from the function call it's
          * the number of elements actually read. Check if numEl before and after the function call differs to know when the end of file is reached.
          * @see  FileDataType.
        **/
        void getSequentialElements(char * dataLine, int   row, int   col, int & numEl);

        /**
          * Gets  the line at position row  from the associated file object and puts it in the
          * character array dataLine.
          * @param  dataLine character array where read data are put.
          * @param row  the line number in the file. If the line is out of bounds then row = -1.
        **/
        void getLine(char * dataLine, int & row);


        /**
          * Initializes the accessor object. The last argument is optional and has a default value of one.
          * @param  filename name of the file to be accessed.
          * @param  filemode access mode of the file.
          * @param endianFile endiannes of the data stored in the file. Values are 'b' or 'B' for big endian and 'l' or 'L' for little endian.
          * @param  type file data type.
          * @param col number of columns of the buffer tile. It must be equal to the number of columns of the associated file.
          * @param row number of rows of the buffer tile. Default is one.
          * @see  getAvailableDataTypes().
          * @see  getAvailableDataTypesAndSizes().
          * @see  AccessMode.
          * @see  FileDataType.
        **/
        void initImageAccessor(string filename, string filemode, char endianFile, string type, int col, int row = 1);

        /**
          * Set the initial line to use getLineSequential().
          * @param  begLine  the initial line. Default is one.
          * @see  getLineSequential().
        **/
        void initSequentialAccessor(int  begLine);


        /**
          * Puts numEl elements in dataLine in the associated file object at the positions column = col[i] and row = row[i]  (for the i-th element).
          * Make sure that the file is already created using createFile() and that the access mode is "readwrite".
          * @param  dataLine character array containing the data.
          * @param  row array with the row positions of the elements to be set.
          * @param col array with the column positions of the elements to be set.
          * @param numEl the number of elements to be set.
          * @see  getElements().
          * @see  createFile().
        **/

        void setElements(char * dataLine, int * row, int * col, int  numEl);

        /**
          * Sets  a line at the position  row.
          * If the full file is not accessed sequentially (i.e. random access), make sure that the file is already created using createFile() and that the access mode is "readwrite".
          * @param  dataLine character array where the data are.
          * @param  row the line number in the file.
        **/
        void setLine(char * dataLine, int  row);

        /** For each call it sets a line from the dataLine character array to the associated file object starting from a given line. The starting line is
          * set using initSequentialAccessor(). The default starting line is one.
          * @param  dataLine character array containing the data to be set.
          * @see  getLineSequential().
          * @see  initSequentialAccessor().
        **/
        void setLineSequential(char * dataLine);
        /**
          * Sets  numEl elements from the character array dataLine to the associated file object starting from the position column = col and row = row.
          * If the full file is not accessed sequentially (i.e. random access), make sure that the file is already created using createFile() and that the access mode is "readwrite".
          * @param  dataLine character array where the data are.
          * @param  row the row position in the file.
          * @param col the  column position int the file.
          * @param numEl  the number of elements to be set.
        **/
        void setSequentialElements(char * dataLine, int  row, int  col, int  numEl);
};
#endif //ImageAccessor_h
