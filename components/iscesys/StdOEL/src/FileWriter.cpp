#include "FileWriter.h" 
#include <cmath>
#include <complex>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
using namespace std;


//public


void FileWriter::initWriter()
{
   FileStream.open(Filename.c_str());
   if(!FileStream)
   {
       cerr << "Error. Cannot open file " << Filename << "." << endl;
       ERR_MESSAGE;
   } 
}
void FileWriter::finalizeWriter()
{ 
    if(FileStream.is_open())
    {
        FileStream.close();
    }
    else
    {
        cerr << "Warining. Attempting to close the not opened file " << Filename  << "." << endl; 
    }
}

void FileWriter::write(string message)
{
    time_t now;
    struct tm * timeInfo;
    time(&now);
    timeInfo = localtime(&now);
    if(FileStream.is_open())
    {
        string tmpStr = asctime(timeInfo);
        size_t pos = tmpStr.find('\n');
        tmpStr.resize(pos);
        if(IncludeTimeStamp)
        {
            FileStream << (FileTag == "" ? FileTag  : FileTag + " : " ) << tmpStr << " : " << message << endl;
        }
        else
        {
            FileStream << (FileTag == "" ? FileTag  : FileTag + " : " )  << message << endl;

        }
    }
    else
    {
        cerr << "Error. Cannot open file " << Filename << "." << endl;
        ERR_MESSAGE;
    }
}
