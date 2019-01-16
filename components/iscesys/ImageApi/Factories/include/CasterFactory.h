#ifndef CasterFactory_h
#define CasterFactory_h

#ifndef MESSAGE
#define MESSAGE cout << "file " << __FILE__ << " line " << __LINE__ << endl;
#endif
#ifndef ERR_MESSAGE
#define ERR_MESSAGE cout << "Error in file " << __FILE__ << " at line " << __LINE__  << " Exiting" <<  endl; exit(1);
#endif

#include <iostream>
#include <stdlib.h>
#include "DataCaster.h"
#include <string>
#include "DoubleToFloatCaster.h"
#include "DoubleToFloatCpxCaster.h"
#include "DoubleToIntCaster.h"
#include "DoubleToIntCpxCaster.h"
#include "DoubleToLongCaster.h"
#include "DoubleToLongCpxCaster.h"
#include "DoubleToShortCaster.h"
#include "DoubleToShortCpxCaster.h"
#include "FloatToDoubleCaster.h"
#include "FloatToDoubleCpxCaster.h"
#include "FloatToIntCaster.h"
#include "FloatToIntCpxCaster.h"
#include "FloatToLongCaster.h"
#include "FloatToLongCpxCaster.h"
#include "FloatToShortCaster.h"
#include "FloatToShortCpxCaster.h"
#include "FloatToByteCaster.h"
#include "IntToDoubleCaster.h"
#include "IntToDoubleCpxCaster.h"
#include "IntToFloatCaster.h"
#include "IntToFloatCpxCaster.h"
#include "IntToLongCaster.h"
#include "IntToLongCpxCaster.h"
#include "IntToShortCaster.h"
#include "IntToShortCpxCaster.h"
#include "LongToDoubleCaster.h"
#include "LongToDoubleCpxCaster.h"
#include "LongToFloatCaster.h"
#include "LongToFloatCpxCaster.h"
#include "LongToIntCaster.h"
#include "LongToIntCpxCaster.h"
#include "LongToShortCaster.h"
#include "LongToShortCpxCaster.h"
#include "ShortToDoubleCaster.h"
#include "ShortToDoubleCpxCaster.h"
#include "ShortToFloatCaster.h"
#include "ShortToFloatCpxCaster.h"
#include "ShortToIntCaster.h"
#include "ShortToIntCpxCaster.h"
#include "ShortToLongCaster.h"
#include "ShortToLongCpxCaster.h"
#include "ByteToFloatCaster.h"
#include "IQByteToFloatCpxCaster.h"
using namespace std;

class CasterFactory
{
    public:
        CasterFactory(){}
        ~CasterFactory(){}
        void printAvailableCasters();
        DataCaster * createCaster(string sel);
    private:
};

#endif //CasterFactory_h
