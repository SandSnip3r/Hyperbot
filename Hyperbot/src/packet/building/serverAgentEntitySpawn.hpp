#ifndef PACKET_BUILDING_SERVER_AGENT_ENTITY_SPAWN_HPP_
#define PACKET_BUILDING_SERVER_AGENT_ENTITY_SPAWN_HPP_

#include "entity/entity.hpp"
#include "packet/opcode.hpp"

#include "../../shared/silkroad_security.h"

namespace packet::building {

// This function was written for a hack and DOES NOT have full functionality
// TODO: Finish
class ServerAgentEntitySpawn {
private:
  static const Opcode kOpcode_ = Opcode::kServerAgentEntitySpawn;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  struct Input {
    sro::scalar_types::EntityGlobalId globalId;
    sro::Position srcPos;
    sro::Angle angle;
    sro::RegionId destinationRegionId;
    int16_t destinationX, destinationY, destinationZ;
    entity::MotionState motionState;
    float walkSpeed, runSpeed, hwanSpeed;
    uint8_t buffCount;
    sro::scalar_types::ReferenceObjectId buffRefId;
    uint32_t buffToken;
  };

  static PacketContainer packet(const Input &input);
};

} // namespace packet::building

#endif // PACKET_BUILDING_SERVER_AGENT_ENTITY_SPAWN_HPP_