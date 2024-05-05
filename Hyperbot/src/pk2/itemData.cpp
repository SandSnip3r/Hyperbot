#include "itemData.hpp"

#include <algorithm>

namespace pk2 {

void ItemData::addItem(ref::Item &&item) {
  items_.emplace(item.id, item);
}

bool ItemData::haveItemWithId(ref::ItemId id) const {
  return (items_.find(id) != items_.end());
}

const ref::Item& ItemData::getItemById(ref::ItemId id) const {
  auto it = items_.find(id);
  if (it == items_.end()) {
    throw std::runtime_error("ItemData::getItemById Trying to get non-existent item with id "+std::to_string(id));
  }
  return it->second;
}

const ref::Item& ItemData::getItemByCodeName128(const std::string &codeName) const {
  auto it = std::find_if(items_.begin(), items_.end(), [&codeName](const auto &keyValuePair) {
    return (keyValuePair.second.codeName128 == codeName);
  });
  if (it == items_.end()) {
    throw std::runtime_error("ItemData::getItemByCodeName128 Item with codename "+codeName+" does not exist");
  }
  return it->second;
}

const ItemData::ItemMap::size_type ItemData::size() const {
  return items_.size();
}

} // namespace pk2