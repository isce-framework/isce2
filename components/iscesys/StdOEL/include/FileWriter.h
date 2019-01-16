#ifndef FileWriter_h
#define FileWriter_h

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
class FileWriter : public BaseWriter
{
    public:
        /// Consrtuctor

    FileWriter()
        {
        }
        /// Destructor
        virtual ~FileWriter()
        {
        if(FileStream.is_open())
        {
            FileStream.close();
        }
        }

    virtual void write(string message);
    virtual void initWriter();
    virtual void finalizeWriter();

    private:

    ofstream FileStream;
        //variables
    //Filename is defined in the base class
};
#endif //FileWriter_h
