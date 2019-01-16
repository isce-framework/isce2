#include "StdOE.h" 
#include <iostream>
#include <fstream>
#include <string>
//initialize defaults. needed to put in a different file. if the defaults were in the StdOE.cpp file, for some reason in going from python-C-fortran-C-C++ it would reinitialize the variables.
string StdOE::FilenameErr;
string StdOE::FilenameLog;
string StdOE::FilenameOut;
string StdOE::FileOutTag;
string StdOE::FileLogTag;
string StdOE::FileErrTag;
ofstream StdOE::FileOut;
ofstream StdOE::FileLog;
ofstream StdOE::FileErr;
char StdOE::StdOut = 's';
char StdOE::StdErr = 's';
