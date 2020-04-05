#ifndef STORAGE_BUYBACK_QUEUE_HPP_
#define STORAGE_BUYBACK_QUEUE_HPP_

#include "item.hpp"

#include <memory>
#include <array>

namespace storage {

class BuybackQueue {
public:
  void addItem(std::shared_ptr<Item> item);
  uint8_t size() const;
  bool hasItem(uint8_t slot) const;
  Item* getItem(uint8_t slot);
  const Item* getItem(uint8_t slot) const;
  std::shared_ptr<Item> withdrawItem(uint8_t slot);

  // Storage() = default;
  // Storage(uint8_t size);
  // bool hasItem(uint8_t slot) const;
  // Item* getItem(uint8_t slot);
  // const Item* getItem(uint8_t slot) const;
  // void resize(uint8_t newSize);
  // void deleteItem(uint8_t slot);
  // void moveItem(uint8_t srcSlot, uint8_t destSlot, uint16_t quantity);
  // void moveItem(uint8_t srcSlot, uint8_t destSlot);
private:
  uint8_t size_{0};
  std::array<std::shared_ptr<Item>, 5> items_;
  void boundsCheck(uint8_t slot, const std::string &funcName) const;
  void destroyItem(uint8_t slot);
};

} // namespace storage

#endif // STORAGE_BUYBACK_QUEUE_HPP_