#ifndef InterleavedFactory_h
#define InterleavedFactory_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif

#include <iostream>
#include <stdlib.h>
#include "InterleavedBase.h"
#include <string>
using namespace std;

class InterleavedFactory
{
    public:
        InterleavedFactory(){}
        ~InterleavedFactory(){}
        InterleavedBase * createInterleaved(string sel);
    private:
};

#endif //InterleavedFactory_h
