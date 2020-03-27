#include "itemList.hpp"

namespace storage {

ItemList::ItemList(uint8_t size) : items_(static_cast<size_t>(size)) {
  //
}

bool ItemList::hasItem(uint8_t slot) const {
  boundsCheck(slot, "hasItem");
  return (items_[slot] != nullptr);
}

item::Item* ItemList::getItem(uint8_t slot) {
  boundsCheck(slot, "getItem");
  return items_[slot].get();
}

const item::Item* ItemList::getItem(uint8_t slot) const {
  boundsCheck(slot, "getItem");
  return items_[slot].get();
}

uint8_t ItemList::size() const {
  return static_cast<uint8_t>(items_.size());
}

void ItemList::addItem(uint8_t slot, std::shared_ptr<item::Item> item) {
  boundsCheck(slot, "addItem");
  if (hasItem(slot)) {
    throw std::runtime_error("ItemList::addItem: Item already exists in slot "+std::to_string(slot));
  }
  items_[slot] = item;
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

void ItemList::boundsCheck(uint8_t slot, const std::string &funcName) const {
  if (slot >= items_.size()) {
    throw std::runtime_error("ItemList::"+funcName+": Out of bounds "+std::to_string(slot)+" >= "+std::to_string(items_.size()));
  }
}

} // namespace storage