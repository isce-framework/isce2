#ifndef FilterFactory_h
#define FilterFactory_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif

#include <iostream>
#include <stdlib.h>
#include "Filter.h"
#include <string>
using namespace std;

class FilterFactory
{
    public:
        FilterFactory(){}
        ~FilterFactory(){}
        Filter * createFilter(string filter,int type);
    private:
};

#endif //FilterFactory_h
