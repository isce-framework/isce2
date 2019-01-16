#include "StdOE.h" 
#include "StdOEF.h" 
#include <cmath>
#include <complex>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

void getStdErr_f(char * c)
{
    (*c) = StdOE::getStdErr();
}
void getStdOut_f(char * c)
{
    (*c) = StdOE::getStdOut();
}
void setStdErr_f(char * message, long int len)
{
    string mess = StdOE::getString(message,len);
    StdOE::setStdErr(mess);
}
void setStdErrFileTag_f(char * tag, long int len)
{
    string tagS = StdOE::getString(tag,len);
    StdOE::setStdErrFileTag(tagS);
}
void setStdOutFileTag_f(char * tag, long int len)
{
    string tagS = StdOE::getString(tag,len);
    StdOE::setStdOutFileTag(tagS);
}
void setStdLogFileTag_f(char * tag, long int len)
{
    string tagS = StdOE::getString(tag,len);
    StdOE::setStdLogFileTag(tagS);
}
void setStdErrFile_f(char * message, long int len)
{
    string mess = StdOE::getString(message,len);
    StdOE::setStdErrFile(mess);
}
void setStdLogFile_f(char * message, long int len)
{
    string mess = StdOE::getString(message,len);
    StdOE::setStdLogFile(mess);
}
void setStdOut_f(char * message, long int len)
{
    string mess = StdOE::getString(message,len);
    StdOE::setStdOut(mess);
}
void setStdOutFile_f(char * message, long int len)
{
    string mess = StdOE::getString(message,len);
    StdOE::setStdOutFile(mess);
}
void writeStd_f(char * message, long int len)
{
    string mess = StdOE::getString(message,len);
    StdOE::writeStd(mess);
}
void writeStdOut_f(char * message, long int len)
{
    string mess = StdOE::getString(message,len);
    StdOE::writeStdOut(mess);
}
void writeStdLog_f(char * message, long int len)
{
    string mess = StdOE::getString(message,len);
    StdOE::writeStdLog(mess);
}
void writeStdErr_f(char * message, long int len)
{
    string mess = StdOE::getString(message,len);
    StdOE::writeStdErr(mess);
}
void writeStdFile_f(char * filename, char * message, long int lenf, long int lenm)
{
    string filen = StdOE::getString(filename,lenf);
    string mess = StdOE::getString(message,lenm);
    StdOE::writeStdFile(filen,mess);
}
