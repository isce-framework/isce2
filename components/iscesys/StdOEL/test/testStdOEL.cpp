#include "StdOEL.h" 
#include "BaseWriter.h" 
#include "WriterFactory.h" 
#include <cmath>
#include <complex>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif

using namespace std;

void writeSomething(uint64_t point)
{
    string message = "first message out";
    string where = "out";
    ((StdOEL *) point)->write(message,where);
    where = "err";
    message = "first message err";
    ((StdOEL *) point)->write(message,where);
    where = "log";
    message = "first message log";
    ((StdOEL *) point)->write(message,where);
    message = "second message out";
    where = "out";
    ((StdOEL *) point)->write(message,where);
    where = "err";
    message = "second message err";
    ((StdOEL *) point)->write(message,where);
    where = "log";
    message = "second message log";
    ((StdOEL *) point)->write(message,where);
} 

int main(int argc, char ** argv)
{
    //defaults to out = err -> screen and log -> file
    WriterFactory WF;
    WF.createWriters();
    BaseWriter * outW = WF.getWriter("out");
    BaseWriter * errW = WF.getWriter("err");
    BaseWriter * logW = WF.getWriter("log");
    string filename = "logFile.log";
    logW->setFilename(filename);
    logW->setFileTag("testTag");
    logW->setTimeStampFlag(true);
    logW->initWriter();
    StdOEL stdOel;
    stdOel.setStd(outW,"out");
    stdOel.setStd(errW,"err");
    stdOel.setStd(logW,"log");
    uint64_t point = (uint64_t) &stdOel;
    writeSomething(point);
    
    logW->finalizeWriter();//close file
    // application is in charge of cleaning up
    delete outW;
    delete errW;
    delete logW;
    
}
