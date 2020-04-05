#ifndef PACKET_PARSING_SERVER_AGENT_CHARACTER_DATA_HPP
#define PACKET_PARSING_SERVER_AGENT_CHARACTER_DATA_HPP

#include "parsedPacket.hpp"
#include "../../pk2/itemData.hpp"
#include "../../pk2/skillData.hpp"
#include "../../shared/silkroad_security.h"

#include <cstdint>
#include <map>
#include <memory>

namespace packet::parsing {
  
class ParsedServerAgentCharacterData : public ParsedPacket {
public:
  ParsedServerAgentCharacterData(const PacketContainer &packet, const pk2::ItemData &itemData, const pk2::SkillData &skillData);
  uint32_t refObjId() const;
  uint64_t gold() const;
  uint32_t entityUniqueId() const;
  uint32_t hp() const;
  uint32_t mp() const;
  uint8_t inventorySize() const;
  const std::map<uint8_t, std::shared_ptr<storage::Item>>& inventoryItemMap() const;
private:
  uint32_t refObjId_;
  uint64_t gold_;
  uint32_t entityUniqueId_;
  uint32_t hp_;
  uint32_t mp_;
  uint8_t inventorySize_;
  std::map<uint8_t, std::shared_ptr<storage::Item>> inventoryItemMap_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_CHARACTER_DATA_HPP