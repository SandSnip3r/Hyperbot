#include "commonParsing.hpp"
#include "serverAgentInventoryStorageData.hpp"

namespace packet::parsing {

ParsedServerAgentInventoryStorageData::ParsedServerAgentInventoryStorageData(const PacketContainer &packet, const pk2::ItemData &itemData) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  gold_ = stream.Read<uint64_t>();
  storageSize_ = stream.Read<uint8_t>();
  uint8_t storageItemCount = stream.Read<uint8_t>();

  for (int i=0; i<storageItemCount; ++i) {
    uint8_t slotNum = stream.Read<uint8_t>();
    storageItemMap_.insert(std::pair<uint8_t, std::shared_ptr<storage::Item>>(slotNum, parseGenericItem(stream, itemData)));
  }
}

uint64_t ParsedServerAgentInventoryStorageData::gold() const {
  return gold_;
}

uint8_t ParsedServerAgentInventoryStorageData::storageSize() const {
  return storageSize_;
}

const std::map<uint8_t, std::shared_ptr<storage::Item>>& ParsedServerAgentInventoryStorageData::storageItemMap() const {
  return storageItemMap_;
}

} // namespace packet::parsing