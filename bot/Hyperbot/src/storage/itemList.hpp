#ifndef STORAGE_ITEMLIST_HPP_
#define STORAGE_ITEMLIST_HPP_

#include "item.hpp"
#include "type_id/typeCategory.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace storage {

class ItemList {
public:
  // Non-modifying
  bool hasItem(uint8_t slot) const;
  Item* getItem(uint8_t slot);
  const Item* getItem(uint8_t slot) const;
  uint8_t size() const;
  std::vector<uint8_t> findItemsOfCategory(const std::vector<type_id::TypeCategory> &categories) const;
  std::vector<uint8_t> findItemsWithTypeId(type_id::TypeId typeId) const;
  std::vector<uint8_t> findItemsWithRefId(uint32_t refId) const;
  std::optional<uint8_t> firstFreeSlot() const;

  // Modifying
  void addItem(uint8_t slot, std::shared_ptr<Item> item);
  std::shared_ptr<Item> withdrawItem(uint8_t slot);
  void moveItem(uint8_t srcSlot, uint8_t destSlot);
  void swapItems(uint8_t srcSlot, uint8_t destSlot);
  void deleteItem(uint8_t slot);
  void resize(uint8_t newSize);
  void clear();
private:
  std::vector<std::shared_ptr<Item>> items_;
  void boundsCheck(uint8_t slot, const std::string &funcName) const;
};

} // namespace storage

#endif // STORAGE_ITEMLIST_HPP_