//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                                  Giangi Sacco
//                        NASA Jet Propulsion Laboratory
//                      California Institute of Technology
//                        (C) 2009  All Rights Reserved
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#ifndef StdOEF_h
#define StdOEF_h
#include "StdOEFFortTrans.h"

/**
  * @file
  * This is a C interface that allows fortran code to call public methods of a StdOE object.

  * The functions name in fortran will be the same except for the suffix "_f" that needs to be removed.
  * Moreover each function "func(args)" will be invoked from fortran using the syntax: call func(args).
  * The correspondence between C and fortran data types is:
  *     - char * <--> character*X (X integer number).
  * @see LineAccessor.cpp
**/
extern "C"
{
    /**
     * Sets (*stdErr) to the value of StdOE::StdErr, i.e. device where the standard error is ridirected.

     @see StdOE::StdErr.
     **/
    void getStdErr_f(char * stdErr);
    /**
     * Sets (*stdOut) to the value of StdOE::StdOut, i.e. device where the standard output is ridirected.

     @see StdOE::StdOut.
     **/
    void getStdOut_f(char * stdOut);

    /**
     * Sets the standard error device. The default is screen.
     @param stdErr  standard error device i.e. file or screen.
     @param length is the length of the string stdErr and is an implicit parameter that does not need to be specified in the fortran function call.
     @see StdOE::StdErr.

     **/
    void setStdErr_f(char * stdErr, long int length);

    /**
     *  Sets a tag that precedes the date in the log file.
     @param tag  string containing the tag to prepend to the date and log message.
     @param length is the length of the string tag and is an implicit parameter that does not need to be specified in the fortran function call.
     @see setStdLogFile_f().
     @see writeStdLog_f().
     **/
    void setStdLogFileTag_f(char * tag  , long int length);

    /**
     *  Sets a tag that precedes the date in the standard output file if the output device is a file.
     @param tag  string containing the tag to prepend to the date and output message.
     @param length is the length of the string tag and is an implicit parameter that does not need to be specified in the fortran function call.
     @see setStdOutFile_f().
     @see setStdOut_f().
     @see writeStdOut_f().
     **/
    void setStdOutFileTag_f(char * tag  , long int length);
    /**
     *  Sets a tag that precedes the date in the standard output file if the output device is a file.
     @param tag  string containing the tag to prepend to the date and output message.
     @param length is the length of the string tag and is an implicit parameter that does not need to be specified in the fortran function call.
     @see setStdErrFile_f().
     @see setStdErr_f().
     @see writeStdErr_f().
     **/
    void setStdErrFileTag_f(char * tag  , long int length);

    /**
     * Sets the name of the file where the standard error is redirected. SdtOE::StdErr is set automatically to 'f', i.e. file.
     @param stdErrFile  standard error filename.
     @param length is the length of the string stdErr and is an implicit parameter that does not need to be specified in the fortran function call.
     @see StdOE::StdErr.

     **/

    void setStdErrFile_f(char * stdErrFile  , long int length);
    /**
     * Sets the name of the file where the log is redirected.
     @param stdLogFile  standard log filename.
     @param length is the length of the string stdLog and is an implicit parameter that does not need to be specified in the fortran function call.
     @see StdOE::StdLog.

     **/

    void setStdLogFile_f(char * stdLogFile  , long int length);

    /**
     * Sets the standard output device. The default is screen.
     @param stdOut standard output device i.e. file or screen.
     @param length is the length of the string stdErr and is an implicit parameter that does not need to be specified in the fortran function call.
     @see StdOE::StdOut.
     **/
    void setStdOut_f(char * stdOut, long int length);

    /**
     * Sets the name of the file where the standard output is redirected. StdOut is set automatically to 'f', i.e. file.
     @param stdOutFile  standard output filename.
     @param length is the length of the string stdErr and is an implicit parameter that does not need to be specified in the fortran function call.
     @see StdOE::StdOut.

        **/
    void setStdOutFile_f(char * stdOutFile, long int length);
    /**
     * Writes the string message on screen.
     @param  message  string to be displayed on screen.
     @param length is the length of the string stdErr and is an implicit parameter that does not need to be specified in the fortran function call.

     **/

    void writeStd_f(char * message, long int length);
    /**
     * Writes the string message in the log file StdOE:FilenameLog.
     *The message  is appended at the end and preceeded by the date in the format Www Mmm dd hh:mm:ss yyyy (see asctime() C++ function documentation).
     @param  message  string to be written on the log file StdOE:FilenameLog.
     @see asctime()
    **/

    void writeStdLog_f(char * message, long int length);
    /**
     * Writes the string message on the preselected standard error device. If the device is a file,
     * it is appended at the end and preceeded by the date in the format Www Mmm dd hh:mm:ss yyyy (see asctime() C++ function documentation).
     @param  message  string to be written on the standard error device.
     @see asctime()
    **/

    void writeStdErr_f(char * message, long int length);
    /**
     * Writes the string message in the file "filename".
     * The message is appended at the end and preceeded by the date in the format Www Mmm dd hh:mm:ss yyyy (see asctime() C++ function documentation).
     @param  filename  name of the file where the string is written.
     @param  message  string to be written into the file.
     @see asctime()

     **/

    void writeStdFile_f(char * filename, char * message, long int length1, long int length2);
    /**
     * Writes the string message on the preselected standard output device. If the device is a file,
     * it is appended at the end and preceeded by the date in the format Www Mmm dd hh:mm:ss yyyy (see asctime() C++ function documentation).
     @param  message  string to be written on the standard error device.
     @see asctime()

     **/

    void writeStdOut_f(char * message, long int length);
}
#endif //StdOEF_h
