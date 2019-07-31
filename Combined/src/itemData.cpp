#include "itemData.hpp"

namespace pk2::media {

void ItemData::addItem(Item &&item) {
  items_.emplace(item.id, item);
}

bool ItemData::haveItemWithId(ItemId id) const {
  return (items_.find(id) != items_.end());
}

const Item& ItemData::getItemById(ItemId id) const {
  auto it = items_.find(id);
  if (it == items_.end()) {
    throw std::runtime_error("Trying to get non-existent item with id "+std::to_string(id));
  }
  return it->second;
}

const ItemData::ItemMap::size_type ItemData::size() const {
  return items_.size();
}

} // namespace pk2::media