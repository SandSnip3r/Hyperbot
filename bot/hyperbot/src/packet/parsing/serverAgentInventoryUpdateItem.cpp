#include "serverAgentInventoryUpdateItem.hpp"

#include <absl/log/log.h>

namespace packet::parsing {

namespace {

template<typename EnumType>
bool hasFlagSet(const EnumType bitmask, const EnumType flag) {
  using RawType = std::underlying_type_t<EnumType>;
  return (static_cast<RawType>(bitmask) & static_cast<RawType>(flag));
}

} // anonymous namespace

ServerAgentInventoryUpdateItem::ServerAgentInventoryUpdateItem(const PacketContainer &packet) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  slotIndex_ = stream.Read<uint8_t>();
  itemUpdateFlag_ = static_cast<enums::ItemUpdateFlag>(stream.Read<std::underlying_type_t<enums::ItemUpdateFlag>>());
  if (hasFlagSet(itemUpdateFlag_, enums::ItemUpdateFlag::kRefObjID)) {
    uint32_t refObjId = stream.Read<uint32_t>();
    LOG(INFO) << "Item at slot " << static_cast<int>(slotIndex_) << " updated ref obj id: " << refObjId;
  }
  if (hasFlagSet(itemUpdateFlag_, enums::ItemUpdateFlag::kOptLevel)) {
    uint8_t optLevel = stream.Read<uint8_t>();
    LOG(INFO) << "Item at slot " << static_cast<int>(slotIndex_) << " updated opt level: " << static_cast<int>(optLevel);
  }
  if (hasFlagSet(itemUpdateFlag_, enums::ItemUpdateFlag::kVariance)) {
    uint64_t variance = stream.Read<uint64_t>();
    LOG(INFO) << "Item at slot " << static_cast<int>(slotIndex_) << " updated variance: " << variance;
  }
  if (hasFlagSet(itemUpdateFlag_, enums::ItemUpdateFlag::kQuantity)) {
    quantity_ = stream.Read<uint16_t>();
  }
  if (hasFlagSet(itemUpdateFlag_, enums::ItemUpdateFlag::kDurability)) {
    uint32_t durability = stream.Read<uint32_t>();
    LOG(INFO) << "Item at slot " << static_cast<int>(slotIndex_) << " updated durability: " << durability;
  }
  if (hasFlagSet(itemUpdateFlag_, enums::ItemUpdateFlag::kMagParams)) {
    uint8_t magParamCount = stream.Read<uint8_t>();
    LOG(INFO) << "Item at slot " << static_cast<int>(slotIndex_) << " updated mag params";
    for (int i=0; i<magParamCount; ++i) {
      uint32_t type = stream.Read<uint32_t>();
      uint32_t value = stream.Read<uint32_t>();
      LOG(INFO) << "  Mag param type " << type;
      LOG(INFO) << "  Mag param value " << value;
    }
  }
  if (hasFlagSet(itemUpdateFlag_, enums::ItemUpdateFlag::kState)) {
    uint8_t state = stream.Read<uint8_t>();
    LOG(INFO) << "Item at slot " << static_cast<int>(slotIndex_) << " updated state: " << static_cast<int>(state);
  }
  if (hasFlagSet(itemUpdateFlag_, enums::ItemUpdateFlag::kUnknown128)) {
    uint32_t unknown = stream.Read<uint32_t>();
    LOG(INFO) << "Item at slot " << static_cast<int>(slotIndex_) << " updated UNKNOWN: " << unknown;
  }
}

uint8_t ServerAgentInventoryUpdateItem::slotIndex() const {
  return slotIndex_;
}

bool ServerAgentInventoryUpdateItem::itemUpdateHasFlag(enums::ItemUpdateFlag flag) const {
  return hasFlagSet(itemUpdateFlag_, flag);
}

uint16_t ServerAgentInventoryUpdateItem::quantity() const {
  return quantity_;
}

// uint8_t ServerAgentInventoryUpdateItem::slotIndex() const {
//   return slotIndex_;
// }

// uint32_t ServerAgentInventoryUpdateItem::durability() const {
//   return durability_;
// }

} // namespace packet::parsing