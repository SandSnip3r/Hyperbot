#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_STATE_HPP
#define PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_STATE_HPP

#include "parsedPacket.hpp"
#include "packet/enums/packetEnums.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <cstdint>

namespace packet::parsing {

enum class PvpState : uint8_t {
  kNeutral = 0, // White
  kAssaulter = 1, // Pink
  kMurderer = 2 // Red
};

enum class BattleState : uint8_t {
  kPeace = 0,
  kBattle = 1
};

enum class ScrollState : uint8_t {
  kNone = 0,
  kReturn = 1, // Unable to move "Resurrect?"
  kThiefDen = 2, // Able to move
};

class ServerAgentEntityUpdateState : public ParsedPacket {
public:
  ServerAgentEntityUpdateState(const PacketContainer &packet);
  sro::scalar_types::EntityGlobalId globalId() const;
  enums::StateType stateType() const;
  uint8_t state() const;
  bool isEnhanced() const;
private:
  sro::scalar_types::EntityGlobalId globalId_;
  enums::StateType stateType_;
  uint8_t state_;
  bool isEnhanced_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_STATE_HPP