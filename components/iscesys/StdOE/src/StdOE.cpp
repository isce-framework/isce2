#include "StdOE.h" 
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


string StdOE::getString(char * word, long int len)
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

void StdOE::setStdErr(string stdErr)
{
    string stdErrCp = stdErr;
    for(int i = 0; i < stdErr.size(); ++i)
    {
        stdErr[i] = tolower(stdErr[i]);
    }
    if(stdErr == "screen")
    {
        StdErr = 's';
    }
    else if(stdErr == "file")
    {
        StdErr = 'f';
    }
    else
    {
        cout << "Unrecognized argument "<< stdErrCp << " in the StdOE constructor." << endl;
        ERR_MESSAGE;
    }
}
void StdOE::setStdErrFileTag(string tag)
{
    FileErrTag = tag;
}
void StdOE::setStdOutFileTag(string tag)
{
    FileOutTag = tag;
}
void StdOE::setStdLogFileTag(string tag)
{
    FileLogTag = tag;
}
void StdOE::setStdErrFile(string stdErrFile)
{
    FilenameErr = stdErrFile;
    StdErr = 'f';
    FileErr.open(FilenameErr.c_str(),ios::app);
    if(!FileErr)
    {
        cout << "Error. Cannot open std error log file "<< FilenameErr << endl;
        ERR_MESSAGE;
    }
}
void StdOE::setStdLogFile(string stdLogFile)
{
    FilenameLog = stdLogFile;
    FileLog.open(FilenameLog.c_str(),ios::app);
    if(!FileLog)
    {
        cout << "Error. Cannot open std  log file "<< FilenameLog << endl;
        ERR_MESSAGE;
    }
}

void StdOE::setStdOut(string stdOut)
{
    string stdOutCp = stdOut;
    for(int i = 0; i < stdOut.size(); ++i)
    {
        stdOut[i] = tolower(stdOut[i]);
    }
    if(stdOut == "screen")
    {
        StdOut = 's';
    }
    else if(stdOut == "file")
    {
        StdOut = 'f';
    }
    else
    {
        cout << "Unrecognized argument "<< stdOutCp << " in the StdOE constructor." << endl;
        ERR_MESSAGE;
    }
}
void StdOE::setStdOutFile(string stdOutFile)
{
    FilenameOut = stdOutFile;
    StdOut = 'f';
    FileOut.open(FilenameOut.c_str(),ios::app);
    if(!FileOut)

    {
        cout << "Error. Cannot open std output log file "<< FilenameOut << endl;
        ERR_MESSAGE;
    }
}

void StdOE::writeStd(string message)
{
    cout <<  message << endl;
}

void StdOE::writeStdErr(string message)
{
    time_t now;
    struct tm * timeInfo;
    time(&now);
    timeInfo = localtime(&now);
    if(StdErr == 's')
    { 
        cout <<  message << endl;
    }
    else if(StdErr == 'f' && FileErr.is_open())
    {
        string tmpStr = asctime(timeInfo);
        size_t pos = tmpStr.find('\n');
        tmpStr.resize(pos);
        FileErr << FileErrTag << " : " << tmpStr << " : " << message << endl;
    }
    else
    {
        cout << "Error. Error log file is not set." << endl;
        ERR_MESSAGE;
    }
}
void StdOE::writeStdLog(string message)
{
    time_t now;
    struct tm * timeInfo;
    time(&now);
    timeInfo = localtime(&now);
    if(FileLog.is_open())
    {
        string tmpStr = asctime(timeInfo);
        size_t pos = tmpStr.find('\n');
        tmpStr.resize(pos);
        FileLog << FileLogTag << " : " <<tmpStr << " : " << message << endl;
    }
    else
    {
        cout << "Error. Log file is not set." << endl;
        ERR_MESSAGE;
    }
}
void StdOE::writeStdFile(string filename,string message)
{
    ofstream fout(filename.c_str(),ios::app);
    if(!fout)
    {
        cout << "Error. Cannot open log file "<< filename << endl;
        ERR_MESSAGE;
    }
    time_t now;
    struct tm * timeInfo;
    time(&now);
    timeInfo = localtime(&now);
    string tmpStr = asctime(timeInfo);
    size_t pos = tmpStr.find('\n');
    tmpStr.resize(pos);
    fout << tmpStr << " : " << message << endl;
}
void StdOE::writeStdOut(string message)
{
    time_t now;
    struct tm * timeInfo;
    time(&now);
    timeInfo = localtime(&now);
    if(StdOut == 's')
    { 
        cout <<  message << endl;
    }
    else if(StdOut == 'f' && FileOut.is_open())
    {
        string tmpStr = asctime(timeInfo);
        size_t pos = tmpStr.find('\n');
        tmpStr.resize(pos);
        FileOut << FileOutTag << " : " <<tmpStr << " : " << message << endl;
    }
    else
    {
        cout << "Error. Output log file is not set." << endl;
        ERR_MESSAGE;
    }
}
