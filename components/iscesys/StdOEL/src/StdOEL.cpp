#include "StdOEL.h" 
#include "BaseWriter.h" 
#include <cmath>
#include <complex>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <cstdio>
#include <stdlib.h>
#include <time.h>
using namespace std;


//public


string StdOEL::getString(char * word, long int len)
{
    int i = len - 1;
    string retStr;
    while(word[i] == ' ')
    {
        --i;
    }
    int count = i;
    while(i >= 0)
    {
        retStr += word[count - i];
        --i;
    }
    return retStr;
}

void StdOEL::setStd(BaseWriter * writer, string type)
{
    Writers[type] = writer;
}

void StdOEL::write(string message, string type)
{
    Writers[type]->write(message);
}
void StdOEL::write_out(string message)
{
    Writers["out"]->write(message);
}
void StdOEL::write_err(string message)
{
    Writers["err"]->write(message);
}
void StdOEL::write_log(string message)
{
    Writers["log"]->write(message);
}
void StdOEL::finalize()
{
    map<string,BaseWriter *>::iterator it;
    for(it = Writers.begin(); it != Writers.end(); ++it)
    {
        it->second->finalizeWriter();
        delete it->second;
    }
}
void StdOEL::init()
{
    map<string,BaseWriter *>::iterator it;
    for(it = Writers.begin(); it != Writers.end(); ++it)
    {
        it->second->initWriter();
    }
}
void StdOEL::setFilename(string filename,string where)
{
    try
    {
        Writers[where]->setFilename(filename);
    }
    catch (exception & e)
    {
        cout << "Error. The Writer of type " << where << " does  not have the method setFilename." << endl;
       ERR_MESSAGE; 
    }
}
void StdOEL::setFileTag(string tag,string where)
{
    try
    {
        map<string,BaseWriter *>::iterator it;
        Writers[where]->setFileTag(tag);
    }
    catch (exception & e)
    {
        cout << "Error. The Writer of type " << where << " does  not have the method setFileTag." << endl;
       ERR_MESSAGE; 
    }
}
void StdOEL::setTimeStampFlag(bool flag ,string where)
{
    try
    {
        Writers[where]->setTimeStampFlag(flag);
    }
    catch (exception & e)
    {
        cout << "Error. The Writer of type " << where << " does  not have the method setTimeStamp." << endl;
       ERR_MESSAGE; 
    }
}
