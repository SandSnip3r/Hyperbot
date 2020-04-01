#ifndef STORAGE_STORAGE_HPP_
#define STORAGE_STORAGE_HPP_

#include "item.hpp"
#include "itemList.hpp"

#include <memory>
#include <string>
#include <vector>

namespace storage {

class Storage {
public:
  Storage() = default;
  Storage(uint8_t size);
  bool hasItem(uint8_t slot) const;
  Item* getItem(uint8_t slot);
  const Item* getItem(uint8_t slot) const;
  uint8_t size() const;
  void clear();
  void resize(uint8_t newSize);
  void addItem(uint8_t slot, std::shared_ptr<Item> item);
  void deleteItem(uint8_t slot);
  void moveItemInStorage(uint8_t srcSlot, uint8_t destSlot, uint16_t quantity);
private:
  ItemList itemList_;
};

} // namespace storage

#endif // STORAGE_STORAGE_HPP_