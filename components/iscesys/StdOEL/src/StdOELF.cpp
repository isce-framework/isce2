#include "StdOEL.h" 
#include "StdOELF.h" 
#include <cmath>
#include <complex>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
using namespace std;

void write_out_f(uint64_t * stdOEL,char * message,long int len)
{
    string mess = ((StdOEL *)(*stdOEL))->getString(message,len);
    string  device = "out";
    ((StdOEL *)(*stdOEL))->write(mess,device);
}
void write_log_f(uint64_t* stdOEL,char * message, long int len)
{
    string mess = ((StdOEL *)(*stdOEL))->getString(message,len);
    string  device = "log";
    ((StdOEL *)(*stdOEL))->write(mess,device);
}
void write_err_f(uint64_t * stdOEL,char * message,long int len)
{
    string mess = ((StdOEL *)(*stdOEL))->getString(message,len);
    string  device = "err";
    ((StdOEL *)(*stdOEL))->write(mess,device);
}
