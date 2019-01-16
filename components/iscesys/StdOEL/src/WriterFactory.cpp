#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <complex>
#include <WriterFactory.h>
#include <map>

using namespace std;

StdOEL * WriterFactory::createStdOEL()
{
    StdOEL * stdOel =  new StdOEL();
    BaseWriter * outW = getWriter("out");
    BaseWriter * errW = getWriter("err");
    BaseWriter * logW = getWriter("log");
    stdOel->setStd(outW,"out");
    stdOel->setStd(errW,"err");
    stdOel->setStd(logW,"log");
    return stdOel;
}
StdOEL *  WriterFactory::createWriters()
{	
    WriterType["out"] = "screen";
    WriterType["err"] = "screen";
    WriterType["log"] = "file";
    return createStdOEL();
}
StdOEL *  WriterFactory::createWriters(string outW)
{	
    WriterType["out"] = outW;
    WriterType["err"] = "screen";
    WriterType["log"] = "file";
    return createStdOEL();
}
StdOEL *  WriterFactory::createWriters(string outW,string errW)
{	
    WriterType["out"] = outW;
    WriterType["err"] = errW;
    WriterType["log"] = "file";
    return createStdOEL();
}
StdOEL *  WriterFactory::createWriters(string outW,string errW, string logW)
{	
    WriterType["out"] = outW;
    WriterType["err"] = errW;
    WriterType["log"] = logW;
    return createStdOEL();
}
BaseWriter * WriterFactory::getWriter(string type)
{
    BaseWriter * retWriter;
    if(WriterType[type] == "file")
    {
        retWriter =  new FileWriter();
    }
    else if(WriterType[type] == "screen")
    {
        retWriter =  new ScreenWriter();
    }
    return retWriter;
}
void WriterFactory::finalize(StdOEL * stdOel)
{
    stdOel->finalize();
    delete stdOel;
}

