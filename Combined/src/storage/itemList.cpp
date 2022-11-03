#include "itemList.hpp"

#include "helpers.hpp"
#include "logging.hpp"

namespace storage {

bool ItemList::hasItem(uint8_t slot) const {
  boundsCheck(slot, "hasItem");
  return static_cast<bool>(items_[slot]);
}

Item* ItemList::getItem(uint8_t slot) {
  boundsCheck(slot, "getItem");
  return items_[slot].get();
}

const Item* ItemList::getItem(uint8_t slot) const {
  boundsCheck(slot, "getItem");
  return items_[slot].get();
}

uint8_t ItemList::size() const {
  return static_cast<uint8_t>(items_.size());
}

std::vector<uint8_t> ItemList::findItemsOfCategory(const std::vector<type_id::TypeCategory> &categories) const {
  std::vector<uint8_t> slotsWithItems;
  for (uint8_t slotNum=0; slotNum<items_.size(); ++slotNum) {
    const auto itemPtr = items_[slotNum];
    if (itemPtr && itemPtr->isOneOf(categories)) {
      // Matches one of the categories
      slotsWithItems.emplace_back(slotNum);
    }
  }
  return slotsWithItems;
}

std::vector<uint8_t> ItemList::findItemsWithTypeId(type_id::TypeId typeId) const {
  const auto [typeId1, typeId2, typeId3, typeId4] = helpers::type_id::splitTypeId(typeId);
  std::vector<uint8_t> slotsWithItem;
  for (uint8_t slotNum=0; slotNum<items_.size(); ++slotNum) {
    const auto itemPtr = items_[slotNum];
    if (itemPtr) {
      const auto &itemRefData = *itemPtr->itemInfo;
      if (itemRefData.typeId1 == typeId1 &&
          itemRefData.typeId2 == typeId2 &&
          itemRefData.typeId3 == typeId3 &&
          itemRefData.typeId4 == typeId4) {
        slotsWithItem.emplace_back(slotNum);
      }
    }
  }
  return slotsWithItem;
}

std::vector<uint8_t> ItemList::findItemsWithRefId(uint32_t refId) const {
  std::vector<uint8_t> slotsWithItem;
  for (uint8_t slotNum=0; slotNum<items_.size(); ++slotNum) {
    const auto itemPtr = items_[slotNum];
    if (itemPtr && itemPtr->refItemId == refId) {
      slotsWithItem.emplace_back(slotNum);
    }
  }
  return slotsWithItem;
}

std::optional<uint8_t> ItemList::firstFreeSlot() const {
  for (uint8_t slotNum=0; slotNum<items_.size(); ++slotNum) {
    const auto itemPtr = items_[slotNum];
    if (itemPtr == nullptr) {
      return slotNum;
    }
  }
  return {};
}

void ItemList::addItem(uint8_t slot, std::shared_ptr<Item> item) {
  boundsCheck(slot, "addItem");
  if (hasItem(slot)) {
    throw std::runtime_error("ItemList::addItem: Item already exists in slot "+std::to_string(slot));
  }
  items_[slot] = item;
}

std::shared_ptr<Item> ItemList::withdrawItem(uint8_t slot) {
  boundsCheck(slot, "withdrawItem");
  if (!hasItem(slot)) {
    throw std::runtime_error("ItemList::withdrawItem: Item doest not exist in slot "+std::to_string(slot));
  }
  auto ptr = items_[slot];
  items_[slot].reset();
  return ptr;
}

void ItemList::moveItem(uint8_t srcSlot, uint8_t destSlot) {
  boundsCheck(srcSlot, "addItem");
  boundsCheck(destSlot, "addItem");
  items_[destSlot] = items_[srcSlot];
  items_[srcSlot].reset();
}

void ItemList::swapItems(uint8_t srcSlot, uint8_t destSlot) {
  boundsCheck(srcSlot, "addItem");
  boundsCheck(destSlot, "addItem");
  std::swap(items_[destSlot], items_[srcSlot]);
}

void ItemList::deleteItem(uint8_t slot) {
  boundsCheck(slot, "deleteItem");
  items_[slot].reset();
}

void ItemList::resize(uint8_t newSize) {
  if (newSize < items_.size()) {
    throw std::runtime_error("ItemList::resize: Trying to reduce size "+std::to_string(items_.size())+" -> "+std::to_string(newSize));
  } else if (newSize > items_.size()) {
    items_.resize(newSize);
  }
}

void ItemList::clear() {
  items_.clear();
}

void ItemList::boundsCheck(uint8_t slot, const std::string &funcName) const {
  if (slot >= items_.size()) {
    throw std::runtime_error("ItemList::"+funcName+": Out of bounds "+std::to_string(slot)+" >= "+std::to_string(items_.size()));
  }
}

} // namespace storage