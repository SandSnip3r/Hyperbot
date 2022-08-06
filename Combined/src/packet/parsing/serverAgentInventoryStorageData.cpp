#include "commonParsing.hpp"
#include "serverAgentInventoryStorageData.hpp"

namespace packet::parsing {

ParsedServerAgentInvetoryStorageData::ParsedServerAgentInvetoryStorageData(const PacketContainer &packet, const pk2::ItemData &itemData) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  gold_ = stream.Read<uint64_t>();
  storageSize_ = stream.Read<uint8_t>();
  uint8_t storageItemCount = stream.Read<uint8_t>();

  for (int i=0; i<storageItemCount; ++i) {
    uint8_t slotNum = stream.Read<uint8_t>();
    auto rentInfo = parseRentInfo(stream);
    uint32_t refItemId = stream.Read<uint32_t>();
    if (!itemData.haveItemWithId(refItemId)) {
      throw std::runtime_error("Unable to parse packet. Encountered an item (id:"+std::to_string(refItemId)+") for which we have no data on.");
    }
    const pk2::ref::Item &itemRef = itemData.getItemById(refItemId);

    std::shared_ptr<storage::Item> parsedItem{storage::newItemByTypeData(itemRef)};
    if (!parsedItem) {
      throw std::runtime_error("Unable to create an item object for item");
    }

    parseItem(parsedItem.get(), stream);
    storageItemMap_.insert(std::pair<uint8_t, std::shared_ptr<storage::Item>>(slotNum, parsedItem));
  }
}

uint64_t ParsedServerAgentInvetoryStorageData::gold() const {
  return gold_;
}

uint8_t ParsedServerAgentInvetoryStorageData::storageSize() const {
  return storageSize_;
}

const std::map<uint8_t, std::shared_ptr<storage::Item>>& ParsedServerAgentInvetoryStorageData::storageItemMap() const {
  return storageItemMap_;
}

} // namespace packet::parsing