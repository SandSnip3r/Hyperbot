#ifndef PACKET_PARSING_SERVER_AGENT_GUILD_STORAGE_DATA_HPP
#define PACKET_PARSING_SERVER_AGENT_GUILD_STORAGE_DATA_HPP

#include "parsedPacket.hpp"
#include "shared/silkroad_security.h"

#include <silkroad_lib/pk2/itemData.hpp>

#include <map>
#include <memory>

namespace packet::parsing {

class ServerAgentGuildStorageData : public ParsedPacket {
public:
  ServerAgentGuildStorageData(const PacketContainer &packet, const sro::pk2::ItemData &itemData);
  uint64_t gold() const;
  uint8_t storageSize() const;
  const std::map<uint8_t, std::shared_ptr<storage::Item>>& storageItemMap() const;
private:
  uint64_t gold_;
  uint8_t storageSize_;
  std::map<uint8_t, std::shared_ptr<storage::Item>> storageItemMap_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_GUILD_STORAGE_DATA_HPP
