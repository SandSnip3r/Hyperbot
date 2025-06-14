#include "commonParsing.hpp"
#include "serverAgentGuildStorageData.hpp"

namespace packet::parsing {

ServerAgentGuildStorageData::ServerAgentGuildStorageData(const PacketContainer &packet, const sro::pk2::ItemData &itemData) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  gold_ = stream.Read<uint64_t>();
  storageSize_ = stream.Read<uint8_t>();
  uint8_t storageItemCount = stream.Read<uint8_t>();

  for (int i=0; i<storageItemCount; ++i) {
    uint8_t slotNum = stream.Read<uint8_t>();
    storageItemMap_.insert(std::pair<uint8_t, std::shared_ptr<storage::Item>>(slotNum, parseGenericItem(stream, itemData)));
  }
}

uint64_t ServerAgentGuildStorageData::gold() const {
  return gold_;
}

uint8_t ServerAgentGuildStorageData::storageSize() const {
  return storageSize_;
}

const std::map<uint8_t, std::shared_ptr<storage::Item>>& ServerAgentGuildStorageData::storageItemMap() const {
  return storageItemMap_;
}

} // namespace packet::parsing