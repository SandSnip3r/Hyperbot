#include "storage.hpp"

#include <algorithm>
#include <iostream>

namespace storage {

namespace {

void printInventorySlot(const storage::ItemList &itemList, uint8_t slot) {
  std::cout << "Slot " << (int)slot << '\n';
  if (!itemList.hasItem(slot)) {
    std::cout << "  Does not exist\n";
    return;
  }
  const item::ItemExpendable *itemExpendable = dynamic_cast<const item::ItemExpendable*>(itemList.getItem(slot));
  if (itemExpendable != nullptr) {
    std::cout << "  x" << itemExpendable->stackCount << '\n';
  } else {
    std::cout << "  Exists\n";
  }
}

}

Storage::Storage(uint8_t size) : itemList_(size) {
  //
}

bool Storage::hasItem(uint8_t slot) const {
  return itemList_.hasItem(slot);
}

item::Item* Storage::getItem(uint8_t slot) {
  return itemList_.getItem(slot);
}

const item::Item* Storage::getItem(uint8_t slot) const {
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

void Storage::addItem(uint8_t slot, std::shared_ptr<item::Item> item) {
  itemList_.addItem(slot, item);
}

void Storage::deleteItem(uint8_t slot) {
  itemList_.deleteItem(slot);
}

void Storage::moveItemInStorage(uint8_t srcSlot, uint8_t destSlot, uint16_t quantity) {
  std::cout << "moveItemInStorage, srcSlot: " << static_cast<int>(srcSlot) << ", destSlot: " << static_cast<int>(destSlot) << ", quantity: " << quantity << '\n';
  std::cout << "Before:\n";
  printInventorySlot(itemList_, srcSlot);
  printInventorySlot(itemList_, destSlot);
  if (itemList_.hasItem(srcSlot)) {
    item::Item *srcItem = itemList_.getItem(srcSlot);
    bool destItemExists = itemList_.hasItem(destSlot);
    if (destItemExists) {
      // Moving into occupied slot
      item::Item *destItem = itemList_.getItem(destSlot);
      if (srcItem->refItemId == destItem->refItemId) {
        // Same type
        item::ItemExpendable *destItemExpendable = dynamic_cast<item::ItemExpendable*>(destItem);
        item::ItemExpendable *srcItemExpendable = dynamic_cast<item::ItemExpendable*>(srcItem);
        if (destItemExpendable != nullptr && srcItemExpendable != nullptr) {
          // Src and dest must be expendable
          // TODO: Check max stack size to see if any are left behind
          const int spaceLeftInStack = destItemExpendable->itemInfo->maxStack - destItemExpendable->stackCount;
          if (spaceLeftInStack == 0) {
            // Is a full stack, just swap
            itemList_.swapItems(srcSlot, destSlot);
          } else {
            // Non-full stack. Add to stack as much as we can
            auto moveCount = std::min(spaceLeftInStack, static_cast<int>(quantity));
            destItemExpendable->stackCount += moveCount;
            if (srcItemExpendable->stackCount > quantity) {
              srcItemExpendable->stackCount -= quantity;
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
      item::ItemExpendable *srcItemExpendable = dynamic_cast<item::ItemExpendable*>(srcItem);
      if (srcItemExpendable != nullptr && srcItemExpendable->stackCount > quantity) {
        // Source is of expendable type and we're splitting
        item::ItemStone *srcItemStone = dynamic_cast<item::ItemStone*>(srcItemExpendable);
        item::ItemMagicPop *srcItemMagicPop = dynamic_cast<item::ItemMagicPop*>(srcItemExpendable);
        // Copy item into slot
        if (srcItemStone != nullptr) {
          // ItemStone
          itemList_.addItem(destSlot, std::shared_ptr<item::Item>(new item::ItemStone(*srcItemStone)));;
        } else if (srcItemMagicPop != nullptr) {
          // ItemMagicPop
          itemList_.addItem(destSlot, std::shared_ptr<item::Item>(new item::ItemMagicPop(*srcItemMagicPop)));;
        } else {
          // ItemExpendable
          itemList_.addItem(destSlot, std::shared_ptr<item::Item>(new item::ItemExpendable(*srcItemExpendable)));;
        }
        // New dest item must be expendable
        item::ItemExpendable *destItemExpendable = dynamic_cast<item::ItemExpendable*>(itemList_.getItem(destSlot));
        destItemExpendable->stackCount = quantity;
        srcItemExpendable->stackCount -= quantity;
      } else {
        // Moving entire stack or non-expendable
        itemList_.moveItem(srcSlot, destSlot);
      }
    }
  } else {
    std::cerr << "Moving an item, but it doesnt exist at the source location\n";
  }
  std::cout << "After:\n";
  printInventorySlot(itemList_, srcSlot);
  printInventorySlot(itemList_, destSlot);
}

} // namespace storage