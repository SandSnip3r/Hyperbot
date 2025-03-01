#ifndef STORAGE_STORAGE_HPP_
#define STORAGE_STORAGE_HPP_

#include "item.hpp"
#include "itemList.hpp"
#include "type_id/typeCategory.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace storage {

namespace detail {

template<typename StoragePtr, typename ItemPtr>
class StorageIterator {
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type        = std::remove_pointer_t<ItemPtr>;
  using difference_type   = std::ptrdiff_t;
  using pointer           = ItemPtr;
  using reference         = value_type&;

  StorageIterator(StoragePtr storage, uint8_t index) : storage_(storage), index_(index) {
    advanceToValid();
  }

  StorageIterator& operator++() {
    ++index_;
    advanceToValid();
    return *this;
  }

  StorageIterator operator++(int) {
    StorageIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  bool operator==(const StorageIterator& other) const {
    return storage_ == other.storage_ && index_ == other.index_;
  }

  bool operator!=(const StorageIterator& other) const {
    return !(*this == other);
  }

  // Now returns a reference to Item
  reference operator*() const {
    return *(storage_->getItem(index_));
  }

  pointer operator->() const {
    return storage_->getItem(index_);
  }

  // Optional: Expose the slot index.
  uint8_t slot() const { return index_; }

private:
  StoragePtr storage_;
  uint8_t index_;

  void advanceToValid() {
    while (index_ < storage_->size() && !storage_->hasItem(index_)) {
      ++index_;
    }
  }
};

}

class Storage {
public:
  using iterator = detail::StorageIterator<Storage*, Item*>;
  using const_iterator = detail::StorageIterator<const Storage*, const Item*>;

  iterator begin() { return iterator(this, 0); }
  iterator end()   { return iterator(this, size()); }

  const_iterator begin() const { return const_iterator(this, 0); }
  const_iterator end()   const { return const_iterator(this, size()); }
  const_iterator cbegin() const { return const_iterator(this, 0); }
  const_iterator cend()   const { return const_iterator(this, size()); }

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

  std::vector<uint8_t> findItemsOfCategory(const std::vector<type_id::TypeCategory> &categories) const;
  std::vector<uint8_t> findItemsWithTypeId(type_id::TypeId typeId) const;
  std::vector<sro::scalar_types::StorageIndexType> findItemsWithRefId(sro::scalar_types::ReferenceObjectId refId) const;
  std::optional<sro::scalar_types::StorageIndexType> findFirstItemWithRefId(sro::scalar_types::ReferenceObjectId refId) const;
  std::optional<uint8_t> firstFreeSlot() const;
private:
  ItemList itemList_;
};

} // namespace storage

#endif // STORAGE_STORAGE_HPP_