#ifndef SRO_STORAGE_H_
#define SRO_STORAGE_H_

#include <silkroad_lib/scalar_types.hpp>

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

  std::string toString() const;
  bool operator==(const Position &other) const;
  bool operator!=(const Position &other) const;

  Storage storage;
  scalar_types::StorageIndexType slotNum;
};

std::string toString(Storage location);

} // namespace sro::storage

#endif // SRO_STORAGE_H_