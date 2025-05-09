#ifndef SRO_STORAGE_H_
#define SRO_STORAGE_H_

#include "scalar_types.hpp"

#include <string>

namespace sro::storage {

enum class Storage {
  kInventory,
  kAvatarInventory,
  kCosInventory,
  kStorage,
  kGuildStorage,
  kNone
};

struct Position {
  Position(Storage storage, scalar_types::StorageIndexType slotNum);
  Storage storage;
  scalar_types::StorageIndexType slotNum;

  bool operator==(const Position &other) const {
    return storage == other.storage && slotNum == other.slotNum;
  }
  bool operator!=(const Position &other) const {
    return !(*this == other);
  }
};

} // namespace sro::storage

std::string toString(sro::storage::Storage location);

#endif // SRO_STORAGE_H_