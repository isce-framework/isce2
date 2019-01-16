#ifndef FactoryWriter_h
#define FactoryWriter_h

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
#include <BaseWriter.h>
#include <ScreenWriter.h>
#include <FileWriter.h>
#include <StdOEL.h>
#include <map>

using namespace std;

/**
    \brief
    * Factory class that provides a selected type of writer.

**/
class WriterFactory
{
    public:
        /// Consrtuctor
    WriterFactory()
    {
    }

    BaseWriter * getWriter(string type);
    StdOEL * createWriters();
        StdOEL * createWriters(string outW);
        StdOEL * createWriters(string outW,string errW);
        StdOEL * createWriters(string outW,string errW, string logW);
    void finalize(StdOEL * stdOel);

        /// Destructor
        ~WriterFactory()
        {
        }


    private:

    StdOEL * createStdOEL();
        //variables
    map<string,string> WriterType;

};
#endif //FactoryWriter_h
