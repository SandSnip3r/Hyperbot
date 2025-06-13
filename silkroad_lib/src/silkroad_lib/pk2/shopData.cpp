#include "shopData.hpp"

#include <absl/log/log.h>

namespace sro::pk2 {

Tab::Tab(const std::string &tabName) : name_(tabName) {}

void Tab::addPackageAtSlot(sro::pk2::ref::ScrapOfPackageItem package, uint8_t slotNum) {
  if (packageMap_.find(slotNum) != packageMap_.end()) {
    LOG(WARNING) << "Warning! Overwriting item in tab " << name_ << " at slot " << (int)slotNum;
  }
  packageMap_[slotNum] = package;
}

bool Tab::havePackage(uint8_t slotNum) const {
  return (packageMap_.find(slotNum) != packageMap_.end());
}

const sro::pk2::ref::ScrapOfPackageItem& Tab::getPackage(uint8_t slotNum) const {
  return packageMap_.at(slotNum);
}

const std::map<uint8_t, sro::pk2::ref::ScrapOfPackageItem>& Tab::getPackageMap() const {
  return packageMap_;
}

const std::string& Tab::getName() const {
  return name_;
}

void ShopData::addTabToNpc(const std::string &npcCodeName, uint8_t tabNum, Tab tab) {
  // Make sure NPC exists
  auto it = npcTabs_.find(npcCodeName);
  if (it == npcTabs_.end()) {
    // Create new NPC with empty Tab list
    auto result = npcTabs_.emplace(npcCodeName, std::vector<Tab>());
    if (!result.second) {
      throw std::runtime_error("Adding NPC "+npcCodeName+" to ShopData failed");
    }
    it = result.first;
  }
  auto &tabList = it->second;
  if (tabList.size() <= tabNum) {
    // Resize Tab list
    tabList.resize(tabNum+1);
  }
  // Save Tab
  tabList[tabNum] = tab;
}

sro::pk2::ref::ScrapOfPackageItem ShopData::getItemFromNpc(const std::string &npcCodeName, uint8_t tabNum, uint8_t slotNum) const {
  const auto it = npcTabs_.find(npcCodeName);
  if (it == npcTabs_.end()) {
    // TODO: Better error handling strategy
    throw std::runtime_error("ShopData::getItemFromNpc Trying to get shop for an npc ("+npcCodeName+") which we have no data on");
  }
  const auto &tabList = it->second;
  if (tabNum >= tabList.size()) {
    // TODO: Better error handling strategy
    throw std::runtime_error("ShopData::getItemFromNpc Tab ("+std::to_string(tabNum)+") out of range for NPC shop ("+npcCodeName+")");
  }
  const auto &tab = tabList.at(tabNum);
  if (!tab.havePackage(slotNum)) {
    // TODO: Better error handling strategy
    throw std::runtime_error("ShopData::getItemFromNpc Slot ("+std::to_string(slotNum)+")  in tab "+std::to_string(tabNum)+" out of range for NPC shop ("+npcCodeName+")");
  }
  return tab.getPackage(slotNum);
}

const std::vector<Tab>& ShopData::getNpcTabs(const std::string &npcCodeName) const {
  const auto it = npcTabs_.find(npcCodeName);
  if (it == npcTabs_.end()) {
    // TODO: Better error handling strategy
    throw std::runtime_error("ShopData::getNpcTabs Trying to get tabs for an npc ("+npcCodeName+") which we have no data on");
  }
  return it->second;
}

} // namespace sro::pk2
