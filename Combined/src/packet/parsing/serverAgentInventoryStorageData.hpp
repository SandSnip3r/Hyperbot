#ifndef PACKET_PARSING_SERVER_AGENT_INVENTORY_STORAGE_DATA_HPP
#define PACKET_PARSING_SERVER_AGENT_INVENTORY_STORAGE_DATA_HPP

#include "parsedPacket.hpp"
#include "../../pk2/itemData.hpp"
#include "../../shared/silkroad_security.h"

#include <map>

namespace packet::parsing {
  
class ParsedServerAgentInvetoryStorageData : public ParsedPacket {
public:
  ParsedServerAgentInvetoryStorageData(const PacketContainer &packet, const pk2::ItemData &itemData);
  uint64_t gold() const;
  uint8_t storageSize() const;
  const std::map<uint8_t, std::shared_ptr<storage::Item>>& storageItemMap() const;
private:
  uint64_t gold_;
  uint8_t storageSize_;
  std::map<uint8_t, std::shared_ptr<storage::Item>> storageItemMap_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_INVENTORY_STORAGE_DATA_HPP