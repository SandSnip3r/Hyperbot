#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_POINTS_HPP
#define PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_POINTS_HPP

#include "parsedPacket.hpp"
#include "packet/enums/packetEnums.hpp"

#include <silkroad_lib/scalar_types.h>

#include <cstdint>
#include <variant>

namespace packet::parsing {

class ServerAgentEntityUpdatePoints : public ParsedPacket {
public:
  ServerAgentEntityUpdatePoints(const PacketContainer &packet);
  packet::enums::UpdatePointsType updatePointsType() const;
  uint64_t gold() const;
  uint32_t skillPoints() const;
  bool isDisplayed() const;
  uint16_t statPoints() const;
  uint8_t hwanPoints() const;
  sro::scalar_types::EntityGlobalId sourceGlobalId() const;
  uint32_t apPoints() const;

private:
  packet::enums::UpdatePointsType updatePointsType_;

  struct GoldUpdate {
    uint64_t gold;
    bool isDisplayed;
  };
  struct SkillPointsUpdate {
    uint32_t skillPoints;
    bool isDisplayed;
  };
  struct StatPointsUpdate {
    uint16_t statPoints;
  };
  struct HwanPointsUpdate {
    uint8_t hwanPoints;
    sro::scalar_types::EntityGlobalId sourceGlobalId;
  };
  struct ApPointsUpdate {
    uint32_t apPoints;
  };
  std::variant<GoldUpdate, SkillPointsUpdate, StatPointsUpdate, HwanPointsUpdate, ApPointsUpdate> updateData_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_POINTS_HPP