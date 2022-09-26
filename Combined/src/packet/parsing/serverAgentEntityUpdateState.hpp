#ifndef PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_STATE_HPP
#define PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_STATE_HPP

#include "parsedPacket.hpp"
#include "../enums/packetEnums.hpp"

#include <silkroad_lib/scalar_types.h>

#include <cstdint>

namespace packet::parsing {

enum class StateType : uint8_t {
  kLifeState = 0,
  kMotionState = 1,
  //2, 3, 5, 6, 9, 10
  kBodyState = 4,
  kPVPState = 7,
  kBattleState = 8,
  kScrollState = 11
};

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
  StateType stateType() const;
  uint8_t state() const;
  bool isEnhanced() const;
private:
  sro::scalar_types::EntityGlobalId globalId_;
  StateType stateType_;
  uint8_t state_;
  bool isEnhanced_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_SERVER_AGENT_ENTITY_UPDATE_STATE_HPP