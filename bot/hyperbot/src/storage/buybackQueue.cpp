#include "buybackQueue.hpp"

#include <algorithm>

namespace storage {

bool BuybackQueue::hasItem(uint8_t slot) const {
  boundsCheck(slot, "hasItem");
  return static_cast<bool>(items_[slot]);
}

uint8_t BuybackQueue::size() const {
  return size_;
}

Item* BuybackQueue::getItem(uint8_t slot) {
  boundsCheck(slot, "getItem");
  if (slot >= size_) {
    throw std::runtime_error("BuybackQueue::getItem Trying to get item "+std::to_string(slot)+" but size is "+std::to_string(size_));
  }
  return items_[slot].get();
}

const Item* BuybackQueue::getItem(uint8_t slot) const {
  boundsCheck(slot, "getItem");
  if (slot >= size_) {
    throw std::runtime_error("BuybackQueue::getItem Trying to get item "+std::to_string(slot)+" but size is "+std::to_string(size_));
  }
  return items_[slot].get();
}

void BuybackQueue::addItem(std::shared_ptr<Item> item) {
  if (size_ == items_.size()) {
    // Queue is full, shift everything over
    destroyItem(0);
  }
  // Assign new item to last free slot
  items_[size_] = item;
  ++size_;
}

std::shared_ptr<Item> BuybackQueue::withdrawItem(uint8_t slot) {
  boundsCheck(slot, "withdrawItem");
  if (slot >= size_) {
    throw std::runtime_error("BuybackQueue::withdrawItem Trying to withdraw item "+std::to_string(slot)+" but size is "+std::to_string(size_));
  }
  auto item = items_[slot];
  // Shift everything over
  destroyItem(slot);
  return item;
}

void BuybackQueue::boundsCheck(uint8_t slot, const std::string &funcName) const {
  if (slot >= items_.size()) {
    throw std::runtime_error("BuybackQueue::boundsCheck Trying to access buyback queue from "+funcName+" beyond bounds, at slot #"+std::to_string(slot));
  }
}

void BuybackQueue::destroyItem(uint8_t slot) {
  for (uint8_t slotNum=slot+1; slotNum<size_; ++slotNum) {
    items_[slotNum-1] = items_[slotNum];
  }
  --size_;
  items_[size_].reset();
}

} // namespace storage