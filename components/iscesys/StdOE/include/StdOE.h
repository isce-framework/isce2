#ifndef StdOE_h
#define StdOE_h

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
#include <cctype>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

using namespace std;

/**
    \brief
    * Class to handle standard output and standar error

    * The class provides a set of convinient methods to write standard output and error on a specified device (screen, file).
    * It consists of static methods and member variables so that they can be used without passing an instance to the calling function.
**/
class StdOE
{
    public:
        /// Consrtuctor
        StdOE()
        {
        }
        /// Destructor
        ~StdOE()
        {
        }
        /**
            * Returns the value of StdErr, i.e. device where the standard error is ridirected.

            @return \c char StdErr.
            @see StdErr.
        **/

        static char getStdErr(){return StdErr;}
        /**
            * Returns the  value of StdOut, i.e. device where the standard output is ridirected.
            @return \c char StdOut.
            @see StdOut.
        **/

        static char getStdOut(){return StdOut;}

        /**
            * Converts a character array received from FORTRAN to a C string.
            @param word character array.
            @param len lenght of the character arrray.
            @return \c string character array in string format.

        **/

        static string getString(char * word, long int len);

        /**
            * Sets the standard error device. The default is screen.
            @param stdErr  standard error device i.e. file or screen.

        **/
        static void setStdErr(string stdErr);

        /**
         *  Sets a tag that precedes the date in the log file.
         @param tag  string to prepend to the date and log message.
         @see setStdLogFile().
         @see writeStdLog().
         **/
        static void setStdLogFileTag(string tag);
        /**
         *  Sets a tag that precedes the date in the standard output file if the output device is a file.
         @param tag  string to prepend to the date and output message.
         @see setStdOutFile().
         @see setStdOut().
         @see writeStdOut().
         **/
        static void setStdErrFileTag(string tag);
        /**
         *  Sets a tag that precedes the date in the standard error file if the output device is a file.
         @param tag  string to prepend to the date and output message.
         @see setStdErrFile().
         @see setStdErr().
         @see writeStdErr().
         **/
        static void setStdOutFileTag(string tag);
        /**
            * Sets the name of the file where the log is redirected.
            @param stdLogFile  log filename.

        **/
        static void setStdLogFile(string stdLogFile);
        /**
            * Sets the name of the file where the standard error is redirected. StdErr is set automatically to 'f', i.e. file.
            @param stdErrFile  standard error filename.
            @see StdErr.

        **/
        static void setStdErrFile(string stdErrFile);
        /**
            * Sets the standard output device. The default is screen.
            @param stdOut standard output device i.e. file or screen.
        **/
        static void setStdOut(string stdOut);
        /**
            * Sets the name of the file where the standard output is redirected. StdOut is set automatically to 'f', i.e. file.
            @param stdOutFile  standard output filename.
            @see StdOut.

        **/

        static void setStdOutFile(string stdOutFile);
        /**
            * Writes the string message on screen.
            @param  message  string to be displayed on screen.

        **/

        static void writeStd(string message);
        /**
            * Writes the string message on the preselected standard error device. If the device is a file,
            * it is appended at the end and preceeded by the date in the format Www Mmm dd hh:mm:ss yyyy (see asctime() C++ function documentation).
            @param  message  string to be written on the standard error device.
            @see asctime()

        **/
        static void writeStdErr(string message);
        /**
            * Writes the string message in the file "filename".
            * The message is appended at the end and preceeded by the date in the format Www Mmm dd hh:mm:ss yyyy (see asctime() C++ function documentation).
            @param  filename  name of the file where the string is written.
            @param  message  string to be written into the file.
            @see asctime()

        **/
        static void writeStdFile(string filename,string message);
        /**
            * Writes the string message on the preselected standard output device. If the device is a file,
            * it is appended at the end and preceeded by the date in the format Www Mmm dd hh:mm:ss yyyy (see asctime() C++ function documentation).
            @param  message  string to be written on the standard error device.
            @see asctime()

        **/
        static void writeStdOut(string message);
        /**
            * Writes the string message in log file FilenameLog.
            * The message is appended at the end and preceeded by the date in the format Www Mmm dd hh:mm:ss yyyy (see asctime() C++ function documentation).
            @param  message  string to be written in the log file FilenameLog.
            @see asctime()

        **/
        static void writeStdLog(string message);


    private:

        //variables

        static ofstream FileErr;
        static ofstream FileOut;
        static ofstream FileLog;
        static string FilenameErr;
        static string FilenameOut;
        static string FilenameLog;
        static string FileOutTag;
        static string FileErrTag;
        static string FileLogTag;
        static char StdOut;
        static char StdErr;
};
#endif //StdOE_h
