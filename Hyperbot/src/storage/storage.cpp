#include "storage.hpp"

#include <absl/log/log.h>

#include <algorithm>

namespace storage {

bool Storage::hasItem(uint8_t slot) const {
  return itemList_.hasItem(slot);
}

Item* Storage::getItem(uint8_t slot) {
  // TODO: Return optional instead. Returning the pointer is dangerous because the item could be moved while the user is referencing it
  return itemList_.getItem(slot);
}

const Item* Storage::getItem(uint8_t slot) const {
  return itemList_.getItem(slot);
}

uint8_t Storage::size() const {
  return itemList_.size();
}

void Storage::clear() {
  itemList_.clear();
}

void Storage::resize(uint8_t newSize) {
  itemList_.resize(newSize);
}

void Storage::addItem(uint8_t slot, std::shared_ptr<Item> item) {
  itemList_.addItem(slot, item);
}

std::shared_ptr<Item> Storage::withdrawItem(uint8_t slot) {
  return itemList_.withdrawItem(slot);
}

void Storage::deleteItem(uint8_t slot) {
  itemList_.deleteItem(slot);
}

void Storage::moveItem(uint8_t srcSlot, uint8_t destSlot, uint16_t quantity) {
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
    LOG(WARNING) << "Moving an item, but it doesnt exist at the source location";
  }
}

void Storage::moveItem(uint8_t srcSlot, uint8_t destSlot) {
  if (itemList_.hasItem(destSlot)) {
    throw std::runtime_error("Storage::moveItem Trying to move an item into a non-open slot");
  }
  if (!itemList_.hasItem(srcSlot)) {
    throw std::runtime_error("Storage::moveItem Trying to move non-existent item");
  }
  itemList_.moveItem(srcSlot, destSlot);
}

std::vector<uint8_t> Storage::findItemsOfCategory(const std::vector<type_id::TypeCategory> &categories) const {
  return itemList_.findItemsOfCategory(categories);
}

std::vector<uint8_t> Storage::findItemsWithTypeId(type_id::TypeId typeId) const {
  return itemList_.findItemsWithTypeId(typeId);
}

std::vector<uint8_t> Storage::findItemsWithRefId(uint32_t refId) const {
  return itemList_.findItemsWithRefId(refId);
}

std::optional<uint8_t> Storage::firstFreeSlot() const {
  return itemList_.firstFreeSlot();
}

} // namespace storage