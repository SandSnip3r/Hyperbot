#include "entity/entity.hpp"
#include "packet/opcode.hpp"
#include "packet/enums/packetEnums.hpp"
#include "packet/structures/packetInnerStructures.hpp"
#include "pk2/characterData.hpp"
#include "pk2/gameData.hpp"
#include "pk2/itemData.hpp"
#include "pk2/skillData.hpp"
#include "pk2/teleportData.hpp"
#include "shared/silkroad_security.h"
#include "storage/item.hpp"

#include <silkroad_lib/position.h>

#include <array>
#include <map>
#include <memory>
#include <string>

#ifndef PACKET_PARSING_HPP
#define PACKET_PARSING_HPP

namespace packet::parsing {

//=========================================================================================================================================================

class ParsedPacket {
protected:
  const Opcode opcode_;
public:
  ParsedPacket(const PacketContainer &packet);
  Opcode opcode() const;
  virtual ~ParsedPacket() = 0;
};

//=========================================================================================================================================================

class ParsedUnknown : public ParsedPacket {
public:
  ParsedUnknown(const PacketContainer &packet);
private:
};

//=========================================================================================================================================================

class ParsedServerAgentEntityUpdateStatus : public ParsedPacket {
public:
  ParsedServerAgentEntityUpdateStatus(const PacketContainer &packet);
  uint32_t entityUniqueId() const;
  packet::enums::UpdateFlag updateFlag() const;
  uint8_t vitalBitmask() const;
  uint32_t newHpValue() const;
  uint32_t newMpValue() const;
  uint16_t newHgpValue() const;
  uint32_t stateBitmask() const;
  const std::vector<uint8_t>& stateLevels() const;
private:
  uint32_t entityUniqueId_;
  packet::enums::UpdateFlag updateFlag_;
  uint8_t vitalBitmask_;
  uint32_t newHpValue_;
  uint32_t newMpValue_;
  uint16_t newHgpValue_;
  uint32_t stateBitmask_;
  std::vector<uint8_t> stateLevels_;
};

//=========================================================================================================================================================

class ParsedServerAgentAbnormalInfo : public ParsedPacket {
public:
  ParsedServerAgentAbnormalInfo(const PacketContainer &packet);
  uint32_t stateBitmask() const;
  const std::array<packet::structures::vitals::AbnormalState, 32>& states() const;
private:
  uint32_t stateBitmask_;
  std::array<packet::structures::vitals::AbnormalState, 32> states_ = {0};
};

//=========================================================================================================================================================

class ParsedServerAgentInventoryItemUseResponse : public ParsedPacket {
public:
  ParsedServerAgentInventoryItemUseResponse(const PacketContainer &packet);
  uint8_t result() const;
  uint8_t slotNum() const;
  uint16_t remainingCount() const;
  uint16_t itemData() const;
  packet::enums::InventoryErrorCode errorCode() const;
private:
  uint8_t result_;
  uint8_t slotNum_;
  uint16_t remainingCount_;
  uint16_t itemData_;
  packet::enums::InventoryErrorCode errorCode_;
};

class ParsedServerAgentCharacterUpdateStats : public ParsedPacket {
public:
  ParsedServerAgentCharacterUpdateStats(const PacketContainer &packet);
  uint32_t maxHp() const;
  uint32_t maxMp() const;
private:
  uint32_t maxHp_;
  uint32_t maxMp_;
};

//=========================================================================================================================================================

enum class GroupSpawnType {
  kSpawn=1,
  kDespawn=2
};

class ParsedServerAgentEntityGroupSpawnData : public ParsedPacket {
public:
  ParsedServerAgentEntityGroupSpawnData(const PacketContainer &packet,
                              const pk2::CharacterData &characterData,
                              const pk2::ItemData &itemData,
                              const pk2::SkillData &skillData,
                              const pk2::TeleportData &teleportData);
  GroupSpawnType groupSpawnType() const;
  const std::vector<std::shared_ptr<entity::Entity>>& entities() const;
  const std::vector<sro::scalar_types::EntityGlobalId>& despawnGlobalIds() const;
private:
  GroupSpawnType groupSpawnType_;
  std::vector<std::shared_ptr<entity::Entity>> entities_;
  std::vector<uint32_t> despawnGlobalIds_;
};

//=========================================================================================================================================================

class ParsedServerAgentSpawn : public ParsedPacket {
public:
  ParsedServerAgentSpawn(const PacketContainer &packet,
                         const pk2::CharacterData &characterData,
                         const pk2::ItemData &itemData,
                         const pk2::SkillData &skillData,
                         const pk2::TeleportData &teleportData);
  std::shared_ptr<entity::Entity> entity() const;
private:
  std::shared_ptr<entity::Entity> entity_;
};

//=========================================================================================================================================================

class ParsedServerAgentDespawn : public ParsedPacket {
public:
  ParsedServerAgentDespawn(const PacketContainer &packet);
  uint32_t globalId() const;
private:
  uint32_t globalId_;
};
  
//=========================================================================================================================================================

class ParsedServerAgentCharacterSelectionJoinResponse : public ParsedPacket {
public:
  ParsedServerAgentCharacterSelectionJoinResponse(const PacketContainer &packet);
  uint8_t result() const;
  uint16_t errorCode() const;
private:
  uint8_t result_;
  uint16_t errorCode_;
};

//=========================================================================================================================================================

class ParsedServerAgentCharacterSelectionActionResponse : public ParsedPacket {
public:
  ParsedServerAgentCharacterSelectionActionResponse(const PacketContainer &packet);
  packet::enums::CharacterSelectionAction action() const;
  uint8_t result() const;
  const std::vector<packet::structures::CharacterSelection::Character>& characters() const;
  uint16_t errorCode() const;
private:
  packet::enums::CharacterSelectionAction action_;
  uint8_t result_;
  std::vector<packet::structures::CharacterSelection::Character> characters_;
  uint16_t errorCode_; // TODO: Create enum for this
};

//=========================================================================================================================================================

class ParsedServerAuthResponse : public ParsedPacket {
public:
  ParsedServerAuthResponse(const PacketContainer &packet);
  uint8_t result() const;
  uint8_t errorCode() const;
private:
  uint8_t result_;
  uint8_t errorCode_;
};

//=========================================================================================================================================================

class ParsedLoginClientInfo : public ParsedPacket {
public:
  ParsedLoginClientInfo(const PacketContainer &packet);
  std::string serviceName() const;
private:
  std::string serviceName_;
};

//=========================================================================================================================================================

class ParsedLoginResponse : public ParsedPacket {
public:
  ParsedLoginResponse(const PacketContainer &packet);
  packet::enums::LoginResult result() const;
  uint32_t token() const;
private:
  packet::enums::LoginResult result_;
  uint32_t token_;
};

//=========================================================================================================================================================

class ParsedLoginServerList : public ParsedPacket {
public:
  ParsedLoginServerList(const PacketContainer &packet);
  uint16_t shardId() const;
private:
  uint16_t shardId_;
};

//=========================================================================================================================================================

class ParsedClientItemMove : public ParsedPacket {
public:
  ParsedClientItemMove(const PacketContainer &packet);
  structures::ItemMovement movement() const;
private:
  structures::ItemMovement movement_;
};

//=========================================================================================================================================================

} // namespace packet::parsing

#endif // PACKET_PARSING_HPP