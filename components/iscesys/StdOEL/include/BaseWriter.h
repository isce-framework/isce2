#ifndef BaseWriter_h
#define BaseWriter_h

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

using namespace std;

/**
    \brief
    * Base abstract class for the writer types.

**/
class BaseWriter
{
    public:
        /// Consrtuctor

    BaseWriter()
        {
        IncludeTimeStamp = false;
        FileTag = "";
        }
        /// Destructor
        virtual ~BaseWriter()
        {
        }

    virtual void write(string message) = 0;
    virtual void initWriter()
    {

    }
    virtual void finalizeWriter()
    {

    }

    void setTimeStampFlag(bool flag)
    {
        IncludeTimeStamp = flag;
    }
    void setFileTag(string tag)
    {
        FileTag = tag;
    }
    void setFilename(string name)
    {
        Filename = name;
    }
    protected:

        //variables
        string FileTag;
    string Filename;
    bool IncludeTimeStamp;

};
#endif //BaseWriter_h
