#ifndef ScreenWriter_h
#define ScreenWriter_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif



#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <complex>
#include <map>
#include <BaseWriter.h>
using namespace std;

/**
    \brief
    * Writer class to write on screen.
    * Derived from BaseWriter()

**/
class ScreenWriter : public BaseWriter
{
    public:
        /// Consrtuctor

    ScreenWriter()
        {
        }
        /// Destructor
        virtual ~ScreenWriter()
        {
        }

    virtual void write(string message);
    private:

        //variables

};
#endif //ScreenWriter_h
