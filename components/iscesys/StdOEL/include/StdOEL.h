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
#include "BaseWriter.h"
#include <map>

using namespace std;

/**
    \brief
    * Class to handle standard output and standar error

    * The class provides a set of convinient methods to write standard output and error on a specified device .
**/
class StdOEL
{
    public:
        /// Consrtuctor
        StdOEL()
        {
        }
        /// Destructor
        ~StdOEL()
        {
        }


    void setFilename(string filename,string where);
    void setFileTag(string tag,string where);
    void setTimeStampFlag(bool flag,string where);
        /**
            * Converts a character array received from FORTRAN to a C string.
            @param word character array.
            @param len lenght of the character arrray.
            @return \c string character array in string format.

        **/

        string getString(char * word, long int len);

        /**
            * Sets the output Object.
            @param writer  pointer to a subclassed BaseWriter.
        @param type type of output. Could be "out", "err" or "log".

        **/
        void setStd(BaseWriter * writer, string type);

        /**
            * Writes the string message on  standard output device.
            @param  message  string to be written on the standard output device.

        **/
        void write_out(string message);
        /**
            * Writes the string message on  standard error device.
            @param  message  string to be written on the standard error device.

        **/
        void write_err(string message);
        /**
            * Writes the string message on  standard log device.
            @param  message  string to be written on the standard log device.

        **/
        void write_log(string message);

        /**
            * Writes the string message on the preselected standard output device.
            @param  message  string to be written on the preselected output device.
        @param type type of output. Could be "out", "err" or "log".

        **/
        void write(string message,string type);


    void finalize();
    void init();

    private:

        //variables
    map<string,BaseWriter *> Writers;
};
#endif //StdOEL_h
