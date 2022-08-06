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

  bool hasItem(uint8_t slot) const;
  Item* getItem(uint8_t slot);
  const Item* getItem(uint8_t slot) const;
  uint8_t size() const;
  
  void clear();
  void resize(uint8_t newSize);
  void addItem(uint8_t slot, std::shared_ptr<Item> item);
  std::shared_ptr<Item> withdrawItem(uint8_t slot);
  void deleteItem(uint8_t slot);
  void moveItem(uint8_t srcSlot, uint8_t destSlot, uint16_t quantity);
  void moveItem(uint8_t srcSlot, uint8_t destSlot);

  std::vector<uint8_t> findItemsWithTypeId(uint16_t typeId) const;
  std::vector<uint8_t> findItemsWithRefId(uint32_t refId) const;
  std::optional<uint8_t> firstFreeSlot() const;
private:
  ItemList itemList_;
};

} // namespace storage

#endif // STORAGE_STORAGE_HPP_