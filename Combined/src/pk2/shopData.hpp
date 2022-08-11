#ifndef PK2_SHOP_DATA_HPP_
#define PK2_SHOP_DATA_HPP_

#include "../../../common/pk2/ref/scrapOfPackageItem.hpp"

#include <map>
#include <string>
#include <vector>
#include <iostream> // TODO: Remove

namespace pk2 {

class Tab {
public:
  Tab() = default;
  Tab(const std::string &tabName);
  void addPackageAtSlot(ref::ScrapOfPackageItem package, uint8_t slotNum);
  bool havePackage(uint8_t slotNum) const;
  const ref::ScrapOfPackageItem& getPackage(uint8_t slotNum) const;
  const std::map<uint8_t, pk2::ref::ScrapOfPackageItem>& getPackageMap() const;
  const std::string& getName() const;
private:
  std::string name_;
  std::map<uint8_t, pk2::ref::ScrapOfPackageItem> packageMap_;
};

class ShopData {
// TODO: Verify that gaps in tabs are impossible
public:
  void addTabToNpc(const std::string &npcCodeName, uint8_t tabNum, Tab tab);
  ref::ScrapOfPackageItem getItemFromNpc(const std::string &npcCodeName, uint8_t tabNum, uint8_t slotNum) const;
  const std::vector<Tab>& getNpcTabs(const std::string &npcCodeName) const;
private:
  std::map<std::string, std::vector<Tab>> npcTabs_;
};

} // namespace pk2

#endif // PK2_SHOP_DATA_HPP_