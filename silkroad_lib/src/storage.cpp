#include "storage.hpp"

namespace sro::storage {

Position::Position(Storage storage, scalar_types::StorageIndexType slotNum)
    : storage(storage), slotNum(slotNum) {}

} // namespace sro::storage

std::string toString(sro::storage::Storage location) {
  switch (location) {
    case sro::storage::Storage::kInventory:
      return "Inventory";
    case sro::storage::Storage::kAvatarInventory:
      return "Avatar Inventory";
    case sro::storage::Storage::kCosInventory:
      return "Cos Inventory";
    case sro::storage::Storage::kStorage:
      return "Storage";
    case sro::storage::Storage::kGuildStorage:
      return "Guild Storage";
    case sro::storage::Storage::kNone:
      return "None";
    default:
      return "Unknown Storage";
  }
}