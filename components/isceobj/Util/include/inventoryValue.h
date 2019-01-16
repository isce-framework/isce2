#ifndef _INVENTORYVALUEMODULE_H_
#define _INVENTORYVALUEMODULE_H_

#include <string>
#include "inventoryValueFortTrans.h"
#include "mroipac/Inventory.h"

extern "C"
{
  void inventoryValChar(Inventory*,char*,char*,int,int);
  void inventoryValNum(Inventory*,char*,char*,void*,int,int);
  void inventoryValNum2(Inventory*,char*,char*,void*,void*,int,int);
  void inventoryValNum3(Inventory*,char*,char*,void*,void*,void*,int,int);
  void inventoryValNum4(Inventory*,char*,char*,void*,void*,void*,void*,int,int);
  void inventoryValArray(Inventory*,char*,char*,void*,int*,int,int);
}

struct InventoryValueError
{
  std::string message;
  InventoryValueError(std::string m){ message = m; }
};


#endif
