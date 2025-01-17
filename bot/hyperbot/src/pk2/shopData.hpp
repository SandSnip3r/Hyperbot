#ifndef PK2_SHOP_DATA_HPP_
#define PK2_SHOP_DATA_HPP_

#include <silkroad_lib/pk2/ref/scrapOfPackageItem.h>

#include <map>
#include <string>
#include <vector>

namespace pk2 {

class Tab {
public:
  Tab() = default;
  Tab(const std::string &tabName);
  void addPackageAtSlot(sro::pk2::ref::ScrapOfPackageItem package, uint8_t slotNum);
  bool havePackage(uint8_t slotNum) const;
  const sro::pk2::ref::ScrapOfPackageItem& getPackage(uint8_t slotNum) const;
  const std::map<uint8_t, sro::pk2::ref::ScrapOfPackageItem>& getPackageMap() const;
  const std::string& getName() const;
private:
  std::string name_;
  std::map<uint8_t, sro::pk2::ref::ScrapOfPackageItem> packageMap_;
};

class ShopData {
// TODO: Verify that gaps in tabs are impossible
public:
  void addTabToNpc(const std::string &npcCodeName, uint8_t tabNum, Tab tab);
  sro::pk2::ref::ScrapOfPackageItem getItemFromNpc(const std::string &npcCodeName, uint8_t tabNum, uint8_t slotNum) const;
  const std::vector<Tab>& getNpcTabs(const std::string &npcCodeName) const;
private:
  std::map<std::string, std::vector<Tab>> npcTabs_;
};

} // namespace pk2

#endif // PK2_SHOP_DATA_HPP_