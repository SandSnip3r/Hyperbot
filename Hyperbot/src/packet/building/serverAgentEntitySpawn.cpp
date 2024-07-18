#include "commonBuilding.hpp"
#include "serverAgentEntitySpawn.hpp"
#include "entity/entity.hpp"

#include <silkroad_lib/entity.h>
#include <silkroad_lib/scalar_types.h>

namespace packet::building {

PacketContainer ServerAgentEntitySpawn::packet(const ServerAgentEntitySpawn::Input &input) {
  StreamUtility stream;
  const sro::scalar_types::ReferenceObjectId entityRefId = 0x00000777;
  stream.Write<>(entityRefId);

  stream.Write<uint8_t>(0x44); // scale
  stream.Write<uint8_t>(0); // hwanLevel
  stream.Write<uint8_t>(0); // pvpCape
  stream.Write<uint8_t>(0); // autoInverstExp
  
  stream.Write<uint8_t>(0x6d); // inventorySize
  stream.Write<uint8_t>(0x07); // inventoryItemCount

  // Item 0
  stream.Write<uint32_t>(0x000011a0); // itemRefId
  stream.Write<uint8_t>(0x05); // optLevel
  // Item 1
  stream.Write<uint32_t>(0x0000120c); // itemRefId
  stream.Write<uint8_t>(0x05); // optLevel
  // Item 2
  stream.Write<uint32_t>(0x000011e8); // itemRefId
  stream.Write<uint8_t>(0x05); // optLevel
  // Item 3
  stream.Write<uint32_t>(0x00001254); // itemRefId
  stream.Write<uint8_t>(0x05); // optLevel
  // Item 4
  stream.Write<uint32_t>(0x00001230); // itemRefId
  stream.Write<uint8_t>(0x05); // optLevel
  // Item 5
  stream.Write<uint32_t>(0x00001278); // itemRefId
  stream.Write<uint8_t>(0x05); // optLevel
  // Item 6
  stream.Write<uint32_t>(0x00001014); // itemRefId
  stream.Write<uint8_t>(0x07); // optLevel

  stream.Write<uint8_t>(0x05); // avatarInventorySize
  stream.Write<uint8_t>(0x05); // avatarInventoryItemCount

  // Item 0
  stream.Write<uint32_t>(0x0000249c); // itemRefId
  stream.Write<uint8_t>(0x00); // optLevel
  // Item 1
  stream.Write<uint32_t>(0x0000604a); // itemRefId
  stream.Write<uint8_t>(0x00); // optLevel
  // Item 2
  stream.Write<uint32_t>(0x000064d3); // itemRefId
  stream.Write<uint8_t>(0x00); // optLevel
  // Item 3
  stream.Write<uint32_t>(0x00006054); // itemRefId
  stream.Write<uint8_t>(0x00); // optLevel
  // Item 4
  stream.Write<uint32_t>(0x00005f5b); // itemRefId
  stream.Write<uint8_t>(0x0a); // optLevel

  stream.Write<uint8_t>(0); // hasMask

  stream.Write<>(input.globalId);
  writePosition(stream, input.srcPos);

  stream.Write<>(input.angle);

  stream.Write<uint8_t>(1); // movementHasDestination
  if (input.motionState == entity::MotionState::kWalk) {
    stream.Write<uint8_t>(0); // movementType 0 = walk, 1 = run
  } else {
    stream.Write<uint8_t>(1); // movementType 0 = walk, 1 = run
  }

  stream.Write<>(input.destinationRegionId);

  stream.Write<>(input.destinationX);
  stream.Write<>(input.destinationY);
  stream.Write<>(input.destinationZ);

  stream.Write<>(sro::entity::LifeState::kAlive);
  stream.Write<uint8_t>(0); // obsolete
  stream.Write<>(input.motionState);

  stream.Write<uint8_t>(0); // bodyState 0=None, 1=Hwan, 2=Untouchable, 3=GameMasterInvincible, 5=GameMasterInvisible, 6=Stealth, 7=Invisible

  stream.Write<>(input.walkSpeed);
  stream.Write<>(input.runSpeed);
  stream.Write<>(input.hwanSpeed);

  stream.Write<>(input.buffCount);
  stream.Write<>(input.buffRefId);
  stream.Write<>(input.buffToken);

  // // Buff 0
  // stream.Write<uint32_t>(0x000079e1); // skillRefId
  // stream.Write<uint32_t>(0x0000021f); // token
  // // Buff 1
  // stream.Write<uint32_t>(0x0000741b); // skillRefId
  // stream.Write<uint32_t>(0x00000221); // token  

  const std::string name = "_Nuked_";
  if (name.length() > std::numeric_limits<uint16_t>::max()) {
    throw std::runtime_error("Name is too long");
  }
  stream.Write<uint16_t>(static_cast<uint16_t>(name.length()));
  stream.Write_Ascii(name);

  stream.Write<uint8_t>(0x02); // jobType
  stream.Write<uint8_t>(0x01); // jobLevel
  stream.Write<uint8_t>(0x00); // murderState

  stream.Write<uint8_t>(0); // isRiding
  stream.Write<uint8_t>(0); // inCombat

  stream.Write<uint8_t>(0); // scrollMode
  stream.Write<uint8_t>(0); // interactMode
  stream.Write<uint8_t>(0); // unkByte4

  stream.Write<uint16_t>(0); // guildNameLength
  // no guild name

  stream.Write<uint32_t>(0); // guildId
  stream.Write<uint16_t>(0); // guildMemberNicknameLength
  // no grant name
  stream.Write<uint32_t>(0); // guildLastCrestRev
  stream.Write<uint32_t>(0); // unionId
  stream.Write<uint32_t>(0); // unionLastCrestRev
  stream.Write<uint8_t>(0); // guildIsFriendly
  stream.Write<uint8_t>(0); // guildMemberSiegeAuthority

  stream.Write<uint8_t>(0); // equipmentCooldown
  stream.Write<uint8_t>(0xff); // pkFlag

  stream.Write<uint8_t>(0x04); //???

  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building