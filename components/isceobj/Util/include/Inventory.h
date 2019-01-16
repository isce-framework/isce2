#if !defined(_MROIPAC_INVENTORY_H_)
#define _MROIPAC_INVENTORY_H_

#include <map>
#include <string>

typedef std::string InventoryKey;
typedef std::string InventoryVal;
typedef std::map<InventoryKey,InventoryVal> Inventory;
typedef std::pair<InventoryKey,InventoryVal> InventoryItem;

#endif

