#ifndef DataAccessorF_h
#define DataAccessorF_h

#include "DataAccessorFFortTrans.h"
#include "DataAccessor.h"
#include <string>
#include <stdint.h>

using namespace std;
/**
 * @file
 * This is a C interface that allows fortran code to call public methods of a DataAccessor object.

 * The functions name in fortran will be the same except for the suffix "_f" that needs to be removed.
 * Moreover each function "func(args)" will be invoked from fortran using the syntax: call func(args).
 * The correspondence between C and fortran data types is:
 *      - uint64_t * <--> integer*8.
 *      - char * <--> character*X (X integer number).
 *      - int * <--> integer or integer*4.
 * @see DataAccessor.cpp
 **/
extern "C"
{
  /**
   * Set the initial line to use getLineSequential_f().
   * @param  ptDataAccessor the value (*ptDataAccessor) is the address of the DataAccessor object.
   * @param begLine the value (*begLine) is the initial line. Default is one.
   * @see  getLineSequential_f().
   **/
  void
  initSequentialAccessor_f(uint64_t * ptDataAccessor, int * begLine);

  /**
   * Prints the available data types and their sizes.
   * Does not require that initDataAccessor_f() be called.
   * @param  ptDataAccessor the value (*ptDataAccessor) is the address of the DataAccessor object.
   **/
  /** For each call it sets a line from the dataLine character array to the associated file object starting from a given line. The starting line is
   * set using initSequentialAccessor(). The default starting line is one.
   * @param ptDataAccessor the value (*ptDataAccessor) is the address of the DataAccessor object.
   * @param dataLine character array containing the data to be set.
   * @see  getLineSequential_f().
   * @see  initSequentialAccessor_f().
   **/
  void
  setLineSequential_f(uint64_t * ptDataAccessor, char * dataLine);
  void
  setLineSequentialBand_f(uint64_t * ptDataAccessor, char * dataLine,
      int * band);

  /**
   * Gets  the line at position (*row)  from the associated file object and puts it in the
   * character array dataLine.
   * @param ptDataAccessor the value (*ptDataAccessor) is the address of the DataAccessor object.
   * @param  dataLine character array where read data are put.
   * @param row the value (*row) is the line number in the file. If the line is out of bounds then (*row) = -1.
   **/
  void
  getLine_f(uint64_t * ptDataAccessor, char * dataLine, int * row);
  void
  getLineBand_f(uint64_t * ptDataAccessor, char * dataLine, int * band,
      int * row);
  /**
   * Sets  (*numEl) elements from the associated file object. The first access is at the beginning of the file. All the subsequent accesses are
   * at the next element of the last one previously accessed.
   * @param  dataLine character array where read data are put.
   * @param  numEl at the function call the value (*numEl) is the number of elements to be read.
   * @see  setSteamAtPos_f().
   * @see  getSteamAtPos_f().
   * @see  getSteam_f().
   **/

  void
  setStream_f(uint64_t * ptDataAccessor, char * dataLine, int * numEl);

  /**
   * Sets  (*numEl) elements from the associated file object at position (*pos). The position is in unit of the FileDataType and NOT in bytes.
   * @param  dataLine character array where read data are put.
   * @param  numEl at the function call the value (*numEl) is the number of elements to be read.
   * @see  setSteamAtPos_f().
   * @see  getSteamAtPos_f().
   * @see  getSteam_f().
   * @see  FileDataType.
   **/
  void
  setStreamAtPos_f(uint64_t * ptDataAccessor, char * dataLine, int * pos,
      int * numEl);

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

  void
  getStream_f(uint64_t * ptDataAccessor, char * dataLine, int * numEl);
  /**
   * Gets  (*numEl) elements from the associated file object at position (*pos). The position is in unit of the FileDataType and NOT in bytes.
   * @param  dataLine character array where read data are put.
   * @param  numEl at the function call the value (*numEl) is the number of elements to be read. At the return from the function call it's
   * the number of elements actually read. Check if (*numEl) before and after the function call differs to know when the end of file is reached.
   * @see  getSteamAtPos_f().
   * @see  setSteamAtPos_f().
   * @see  setSteam_F().
   **/

  void
  getStreamAtPos_f(uint64_t * ptDataAccessor, char * dataLine, int * pos,
      int * numEl);

  /**
   * Sets  a line at the position  (*row).
   * If the full file is not accessed sequentially (i.e. random access), make sure that the file is already created using createFile() and that the access mode is "readwrite".
   * @param  ptDataAccessor the value (*ptDataAccessor) is the address of the DataAccessor object.
   * @param  dataLine character array where the data are.
   * @param  row the value (*row) is the line number in the file.
   **/
  void
  setLine_f(uint64_t * ptDataAccessor, char * dataLine, int * row);
  void
  setLineBand_f(uint64_t * ptDataAccessor, char * dataLine, int * row,
      int * band);
  /**
   * For each call it gets a line from the associated file object and puts it in the character array dataLine starting from a given line. The starting
   * line is set using initSequentialAccessor(). The default starting line is one.
   * @param  ptDataAccessor the value (*ptDataAccessor) is the address of the DataAccessor object.
   * @param dataLine character array where read data are put.
   * @param  eof the value (*eof) is set to -1 when the end of file is reached otherwise it give the position of the line just read.
   * @see  setLineSequential_f().
   * @see  initSequentialAccessor_f().
   **/
  void
  getLineSequential_f(uint64_t * ptDataAccessor, char * dataLine, int * eof);
  void
  getLineSequentialBand_f(uint64_t * ptDataAccessor, char * dataLine,
      int * band, int * eof);

  void
  getSequentialElements_f(uint64_t * ptDataAccessor, char * dataLine,
      int * ptRow, int * ptCol, int * ptNumEl);
  void
  setSequentialElements_f(uint64_t * ptDataAccessor, char * dataLine,
      int * ptRow, int * ptCol, int * ptNumEl);
  void
  rewindAccessor_f(uint64_t*);
  double
  getPx2d_f(uint64_t * ptDataAccessor, int * ptRow, int * ptCol);
  double
  getPx1d_f(uint64_t * ptDataAccessor, int * ptPos);
  int
  getWidth_f(uint64_t * ptDataAccessor);
  int
  getNumberOfLines_f(uint64_t * ptDataAccessor);
/**
 * Set the first pixel read for each line
 * @param  ptDataAccessor the value (*ptDataAccessor) is the address of the DataAccessor object.
 * @param lineoff  the first pixel to be read
 **/
 void setLineOffset_f(uint64_t * ptDataAccessor,int * lineoff);
 /**
  * Get the first pixel read for each line
  * @param  ptDataAccessor the value (*ptDataAccessor) is the address of the DataAccessor object.
  * @return the first pixel read for each line
  **/
 int getLineOffset_f(uint64_t * ptDataAccessor);
 }
#endif //DataAccessorF_h
