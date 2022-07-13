#include "storage.hpp"

#include <algorithm>
#include <iostream>

namespace storage {

Storage::Storage(std::mutex &mutex) : storageMutex_(mutex) {}

bool Storage::hasItem(uint8_t slot) const {
  std::unique_lock<std::mutex> storageLock(storageMutex_);
  return itemList_.hasItem(slot);
}

Item* Storage::getItem(uint8_t slot) {
  std::unique_lock<std::mutex> storageLock(storageMutex_);
  // TODO: Return optional instead. Returning the pointer is dangerous because the item could be moved while the user is referencing it
  return itemList_.getItem(slot);
}

const Item* Storage::getItem(uint8_t slot) const {
  std::unique_lock<std::mutex> storageLock(storageMutex_);
  return itemList_.getItem(slot);
}

uint8_t Storage::size() const {
  std::unique_lock<std::mutex> storageLock(storageMutex_);
  return itemList_.size();
}

void Storage::clear() {
  std::unique_lock<std::mutex> storageLock(storageMutex_);
  itemList_.clear();
}

void Storage::resize(uint8_t newSize) {
  std::unique_lock<std::mutex> storageLock(storageMutex_);
  itemList_.resize(newSize);
}

void Storage::addItem(uint8_t slot, std::shared_ptr<Item> item) {
  std::unique_lock<std::mutex> storageLock(storageMutex_);
  itemList_.addItem(slot, item);
}

std::shared_ptr<Item> Storage::withdrawItem(uint8_t slot) {
  std::unique_lock<std::mutex> storageLock(storageMutex_);
  return itemList_.withdrawItem(slot);
}

void Storage::deleteItem(uint8_t slot) {
  std::unique_lock<std::mutex> storageLock(storageMutex_);
  itemList_.deleteItem(slot);
}

void Storage::moveItem(uint8_t srcSlot, uint8_t destSlot, uint16_t quantity) {
  std::unique_lock<std::mutex> storageLock(storageMutex_);
  std::cout << "moveItem, srcSlot: " << static_cast<int>(srcSlot) << ", destSlot: " << static_cast<int>(destSlot) << ", quantity: " << quantity << '\n';
  if (itemList_.hasItem(srcSlot)) {
    Item *srcItem = itemList_.getItem(srcSlot);
    bool destItemExists = itemList_.hasItem(destSlot);
    if (destItemExists) {
      // Moving into occupied slot
      Item *destItem = itemList_.getItem(destSlot);
      if (srcItem->refItemId == destItem->refItemId) {
        // Same type
        ItemExpendable *destItemExpendable = dynamic_cast<ItemExpendable*>(destItem);
        ItemExpendable *srcItemExpendable = dynamic_cast<ItemExpendable*>(srcItem);
        if (destItemExpendable != nullptr && srcItemExpendable != nullptr) {
          // Src and dest must be expendable
          const int spaceLeftInStack = destItemExpendable->itemInfo->maxStack - destItemExpendable->quantity;
          if (spaceLeftInStack == 0) {
            // Is a full stack, just swap
            itemList_.swapItems(srcSlot, destSlot);
          } else {
            // Non-full stack. Add to stack as much as we can
            auto moveCount = std::min(spaceLeftInStack, static_cast<int>(quantity));
            destItemExpendable->quantity += moveCount;
            if (srcItemExpendable->quantity > quantity) {
              srcItemExpendable->quantity -= quantity;
            } else {
              itemList_.deleteItem(srcSlot);
              // srcItem and srcItemExpendable point to garbage now!
            }
          }
        } else {
          // Not expendables, just swapping two items of the same type
          itemList_.swapItems(srcSlot, destSlot);
        }
      } else {
        // Different type, swapping
        itemList_.swapItems(srcSlot, destSlot);
      }
    } else {
      // Moving into open slot
      ItemExpendable *srcItemExpendable = dynamic_cast<ItemExpendable*>(srcItem);
      if (srcItemExpendable != nullptr && srcItemExpendable->quantity > quantity) {
        // Source is of expendable type and we're splitting
        std::shared_ptr<storage::Item> clonedItem(storage::cloneItem(srcItem));
        // // New dest item must be expendable
        ItemExpendable *destItemExpendable = dynamic_cast<ItemExpendable*>(clonedItem.get());
        destItemExpendable->quantity = quantity;
        srcItemExpendable->quantity -= quantity;
        itemList_.addItem(destSlot, clonedItem);
      } else {
        // Moving entire stack or non-expendable
        itemList_.moveItem(srcSlot, destSlot);
      }
    }
  } else {
    std::cerr << "Moving an item, but it doesnt exist at the source location\n";
  }
}

void Storage::moveItem(uint8_t srcSlot, uint8_t destSlot) {
  std::unique_lock<std::mutex> storageLock(storageMutex_);
  if (itemList_.hasItem(destSlot)) {
    throw std::runtime_error("Storage::moveItem Trying to move an item into a non-open slot");
  }
  if (!itemList_.hasItem(srcSlot)) {
    throw std::runtime_error("Storage::moveItem Trying to move non-existent item");
  }
  itemList_.moveItem(srcSlot, destSlot);
}

std::vector<uint8_t> Storage::findItemsWithTypeId(uint8_t typeId1, uint8_t typeId2, uint8_t typeId3, uint8_t typeId4) const {
  // TODO: Implement a function just to return the first
  return itemList_.findItemsWithTypeId(typeId1, typeId2, typeId3, typeId4);
}

} // namespace storage