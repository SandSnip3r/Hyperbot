#ifndef PACKET_PARSING_SERVER_AGENT_COS_DATA_HPP_
#define PACKET_PARSING_SERVER_AGENT_COS_DATA_HPP_

#include "parsedPacket.hpp"
#include "storage/item.hpp"

#include <silkroad_lib/pk2/characterData.hpp>
#include <silkroad_lib/pk2/itemData.hpp>

#include <cstdint>
#include <map>

namespace packet::parsing {
  
class ServerAgentCosData : public ParsedPacket {
public:
  ServerAgentCosData(const PacketContainer &packet, const sro::pk2::CharacterData &characterData, const sro::pk2::ItemData &itemData);
  uint32_t globalId() const;
  bool isAbilityPet() const;
  uint8_t inventorySize() const;
  const std::map<uint8_t, std::shared_ptr<storage::Item>>& inventoryItemMap() const;
  uint32_t ownerGlobalId() const;
private:
  uint32_t globalId_;
  uint8_t typeId4_;
  uint8_t inventorySize_;
  std::map<uint8_t, std::shared_ptr<storage::Item>> inventoryItemMap_;
  uint32_t ownerGlobalId_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_COS_DATA_HPP_
