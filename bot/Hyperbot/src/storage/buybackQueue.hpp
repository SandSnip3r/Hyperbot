#ifndef STORAGE_BUYBACK_QUEUE_HPP_
#define STORAGE_BUYBACK_QUEUE_HPP_

#include "item.hpp"

#include <array>
#include <memory>

namespace storage {

class BuybackQueue {
public:
  void addItem(std::shared_ptr<Item> item);
  uint8_t size() const;
  bool hasItem(uint8_t slot) const;
  Item* getItem(uint8_t slot);
  const Item* getItem(uint8_t slot) const;
  std::shared_ptr<Item> withdrawItem(uint8_t slot);
private:
  uint8_t size_{0};
  std::array<std::shared_ptr<Item>, 5> items_;
  void boundsCheck(uint8_t slot, const std::string &funcName) const;
  void destroyItem(uint8_t slot);
};

} // namespace storage

#endif // STORAGE_BUYBACK_QUEUE_HPP_