#include <silkroad_lib/storage.hpp>

#include <absl/strings/str_format.h>

namespace sro::storage {

Position::Position(Storage storage, scalar_types::StorageIndexType slotNum)
    : storage(storage), slotNum(slotNum) {}

std::string Position::toString() const {
  return absl::StrFormat("%s:%d", sro::storage::toString(storage), slotNum);
}

bool Position::operator==(const Position &other) const {
  return storage == other.storage && slotNum == other.slotNum;
}

bool Position::operator!=(const Position &other) const {
  return !(*this == other);
}

std::string toString(Storage location) {
  switch (location) {
    case Storage::kInventory:
      return "Inventory";
    case Storage::kAvatarInventory:
      return "Avatar Inventory";
    case Storage::kCosInventory:
      return "Cos Inventory";
    case Storage::kStorage:
      return "Storage";
    case Storage::kGuildStorage:
      return "Guild Storage";
    case Storage::kNone:
      return "None";
    default:
      return "Unknown Storage";
  }
}

} // namespace sro::storage