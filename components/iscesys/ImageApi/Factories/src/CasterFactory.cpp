#include "CasterFactory.h"
#include "DataCaster.h"
#include <string>
#include <vector>
using namespace std;

DataCaster *
CasterFactory::createCaster(string sel)
{
  if (sel == "DoubleToFloatCaster")
  {
    return new DoubleToFloatCaster();
  }
  else if (sel == "DoubleToFloatCpxCaster")
  {
    return new DoubleToFloatCpxCaster();
  }
  else if (sel == "DoubleToIntCaster")
  {
    return new DoubleToIntCaster();
  }
  else if (sel == "DoubleToIntCpxCaster")
  {
    return new DoubleToIntCpxCaster();
  }
  else if (sel == "DoubleToLongCaster")
  {
    return new DoubleToLongCaster();
  }
  else if (sel == "DoubleToLongCpxCaster")
  {
    return new DoubleToLongCpxCaster();
  }
  else if (sel == "DoubleToShortCaster")
  {
    return new DoubleToShortCaster();
  }
  else if (sel == "DoubleToShortCpxCaster")
  {
    return new DoubleToShortCpxCaster();
  }
  else if (sel == "FloatToDoubleCaster")
  {
    return new FloatToDoubleCaster();
  }
  else if (sel == "FloatToDoubleCpxCaster")
  {
    return new FloatToDoubleCpxCaster();
  }
  else if (sel == "FloatToIntCaster")
  {
    return new FloatToIntCaster();
  }
  else if (sel == "FloatToIntCpxCaster")
  {
    return new FloatToIntCpxCaster();
  }
  else if (sel == "FloatToLongCaster")
  {
    return new FloatToLongCaster();
  }
  else if (sel == "FloatToLongCpxCaster")
  {
    return new FloatToLongCpxCaster();
  }
  else if (sel == "FloatToShortCaster")
  {
    return new FloatToShortCaster();
  }
  else if (sel == "FloatToShortCpxCaster")
  {
    return new FloatToShortCpxCaster();
  }
  else if (sel == "FloatToByteCaster")
  {
      return new FloatToByteCaster();
  }
  else if (sel == "IntToDoubleCaster")
  {
    return new IntToDoubleCaster();
  }
  else if (sel == "IntToDoubleCpxCaster")
  {
    return new IntToDoubleCpxCaster();
  }
  else if (sel == "IntToFloatCaster")
  {
    return new IntToFloatCaster();
  }
  else if (sel == "IntToFloatCpxCaster")
  {
    return new IntToFloatCpxCaster();
  }
  else if (sel == "IntToLongCaster")
  {
    return new IntToLongCaster();
  }
  else if (sel == "IntToLongCpxCaster")
  {
    return new IntToLongCpxCaster();
  }
  else if (sel == "IntToShortCaster")
  {
    return new IntToShortCaster();
  }
  else if (sel == "IntToShortCpxCaster")
  {
    return new IntToShortCpxCaster();
  }
  else if (sel == "LongToDoubleCaster")
  {
    return new LongToDoubleCaster();
  }
  else if (sel == "LongToDoubleCpxCaster")
  {
    return new LongToDoubleCpxCaster();
  }
  else if (sel == "LongToFloatCaster")
  {
    return new LongToFloatCaster();
  }
  else if (sel == "LongToFloatCpxCaster")
  {
    return new LongToFloatCpxCaster();
  }
  else if (sel == "LongToIntCaster")
  {
    return new LongToIntCaster();
  }
  else if (sel == "LongToIntCpxCaster")
  {
    return new LongToIntCpxCaster();
  }
  else if (sel == "LongToShortCaster")
  {
    return new LongToShortCaster();
  }
  else if (sel == "LongToShortCpxCaster")
  {
    return new LongToShortCpxCaster();
  }
  else if (sel == "ShortToDoubleCaster")
  {
    return new ShortToDoubleCaster();
  }
  else if (sel == "ShortToDoubleCpxCaster")
  {
    return new ShortToDoubleCpxCaster();
  }
  else if (sel == "ShortToFloatCaster")
  {
    return new ShortToFloatCaster();
  }
  else if (sel == "ShortToFloatCpxCaster")
  {
    return new ShortToFloatCpxCaster();
  }
  else if (sel == "ShortToIntCaster")
  {
    return new ShortToIntCaster();
  }
  else if (sel == "ShortToIntCpxCaster")
  {
    return new ShortToIntCpxCaster();
  }
  else if (sel == "ShortToLongCaster")
  {
    return new ShortToLongCaster();
  }
  else if (sel == "ShortToLongCpxCaster")
  {
    return new ShortToLongCpxCaster();
  }
  else if (sel == "ByteToFloatCaster")
  {
    return new ByteToFloatCaster();
  }
  else if (sel == "IQByteToFloatCpxCaster")
  {
    return new IQByteToFloatCpxCaster();
  }
  else
  {
    cout << "Error. " << sel << " is an unrecognized Caster." << endl;
    cout << "Available casters are :" << endl;
    printAvailableCasters();
    ERR_MESSAGE
    ;
  }
}
void
CasterFactory::printAvailableCasters()
{
  vector < string > casterList;
  casterList.push_back("DoubleToFloatCaster");
  casterList.push_back("DoubleToFloatCpxCaster");
  casterList.push_back("DoubleToIntCaster");
  casterList.push_back("DoubleToIntCpxCaster");
  casterList.push_back("DoubleToLongCaster");
  casterList.push_back("DoubleToLongCpxCaster");
  casterList.push_back("DoubleToShortCaster");
  casterList.push_back("DoubleToShortCpxCaster");
  casterList.push_back("FloatToDoubleCaster");
  casterList.push_back("FloatToDoubleCpxCaster");
  casterList.push_back("FloatToIntCaster");
  casterList.push_back("FloatToIntCpxCaster");
  casterList.push_back("FloatToLongCaster");
  casterList.push_back("FloatToLongCpxCaster");
  casterList.push_back("FloatToShortCaster");
  casterList.push_back("FloatToShortCpxCaster");
  casterList.push_back("FloatToByteCaster");
  casterList.push_back("IntToDoubleCaster");
  casterList.push_back("IntToDoubleCpxCaster");
  casterList.push_back("IntToFloatCaster");
  casterList.push_back("IntToFloatCpxCaster");
  casterList.push_back("IntToLongCaster");
  casterList.push_back("IntToLongCpxCaster");
  casterList.push_back("IntToShortCaster");
  casterList.push_back("IntToShortCpxCaster");
  casterList.push_back("LongToDoubleCaster");
  casterList.push_back("LongToDoubleCpxCaster");
  casterList.push_back("LongToFloatCaster");
  casterList.push_back("LongToFloatCpxCaster");
  casterList.push_back("LongToIntCaster");
  casterList.push_back("LongToIntCpxCaster");
  casterList.push_back("LongToShortCaster");
  casterList.push_back("LongToShortCpxCaster");
  casterList.push_back("ShortToDoubleCaster");
  casterList.push_back("ShortToDoubleCpxCaster");
  casterList.push_back("ShortToFloatCaster");
  casterList.push_back("ShortToFloatCpxCaster");
  casterList.push_back("ShortToIntCaster");
  casterList.push_back("ShortToIntCpxCaster");
  casterList.push_back("ShortToLongCaster");
  casterList.push_back("ShortToLongCpxCaster");
  casterList.push_back("IQByteToFloatCpxCaster");

  for (int i = 0; i < casterList.size(); ++i)
  {
    cout << casterList[i] << endl;
  }

}
