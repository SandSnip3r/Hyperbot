#ifndef STORAGE_ITEMLIST_HPP_
#define STORAGE_ITEMLIST_HPP_

#include "item.hpp"

#include <memory>
#include <string>
#include <vector>

namespace storage {

class ItemList {
public:
  ItemList() = default;
  ItemList(uint8_t size);

  // Non-modifying
  bool hasItem(uint8_t slot) const;
  item::Item* getItem(uint8_t slot);
  const item::Item* getItem(uint8_t slot) const;
  uint8_t size() const;

  // Modifying
  void addItem(uint8_t slot, std::shared_ptr<item::Item> item);
  void moveItem(uint8_t srcSlot, uint8_t destSlot);
  void swapItems(uint8_t srcSlot, uint8_t destSlot);
  void deleteItem(uint8_t slot);
  void resize(uint8_t newSize);
private:
  std::vector<std::shared_ptr<item::Item>> items_;
  void boundsCheck(uint8_t slot, const std::string &funcName) const;
};

} // namespace storage

#endif // STORAGE_ITEMLIST_HPP_