#ifndef PACKETPROCESSOR_HPP_
#define PACKETPROCESSOR_HPP_

#include "broker/eventBroker.hpp"
#include "broker/packetBroker.hpp"
// #include "packet/parsing/clientAgentActionDeselectRequest.hpp"
// #include "packet/parsing/clientAgentActionSelectRequest.hpp"
#include "packet/parsing/clientAgentActionCommandRequest.hpp"
#include "packet/parsing/clientAgentActionTalkRequest.hpp"
#include "packet/parsing/serverAgentActionCommandResponse.hpp"
#include "packet/parsing/serverAgentActionDeselectResponse.hpp"
#include "packet/parsing/serverAgentActionSelectResponse.hpp"
#include "packet/parsing/serverAgentActionTalkResponse.hpp"
#include "packet/parsing/serverAgentBuffAdd.hpp"
#include "packet/parsing/serverAgentBuffRemove.hpp"
#include "packet/parsing/serverAgentCharacterData.hpp"
#include "packet/parsing/serverAgentCosData.hpp"
#include "packet/parsing/serverAgentEntityDespawn.hpp"
#include "packet/parsing/serverAgentEntityGroupSpawnData.hpp"
#include "packet/parsing/serverAgentEntityRemoveOwnership.hpp"
#include "packet/parsing/serverAgentEntitySpawn.hpp"
#include "packet/parsing/serverAgentEntitySyncPosition.hpp"
#include "packet/parsing/serverAgentEntityUpdateAngle.hpp"
#include "packet/parsing/serverAgentEntityUpdateExperience.hpp"
#include "packet/parsing/serverAgentEntityUpdateMovement.hpp"
#include "packet/parsing/serverAgentEntityUpdateMoveSpeed.hpp"
#include "packet/parsing/serverAgentEntityUpdatePoints.hpp"
#include "packet/parsing/serverAgentEntityUpdatePosition.hpp"
#include "packet/parsing/serverAgentEntityUpdateState.hpp"
#include "packet/parsing/serverAgentEntityUpdateStatus.hpp"
#include "packet/parsing/serverAgentGuildStorageData.hpp"
#include "packet/parsing/serverAgentInventoryItemUseResponse.hpp"
#include "packet/parsing/serverAgentInventoryOperationResponse.hpp"
#include "packet/parsing/serverAgentInventoryRepairResponse.hpp"
#include "packet/parsing/serverAgentInventoryStorageData.hpp"
#include "packet/parsing/serverAgentInventoryUpdateDurability.hpp"
#include "packet/parsing/serverAgentInventoryUpdateItem.hpp"
#include "packet/parsing/serverAgentSkillBegin.hpp"
#include "packet/parsing/serverAgentSkillEnd.hpp"
#include "packet/parsing/packetParser.hpp"
#include "pk2/gameData.hpp"
#include "state/worldState.hpp"

#define ENFORCE_PURIFICATION_PILL_COOLDOWN

/*  PacketProcessor
 *  As packets come in, this class will update the state
 */
class PacketProcessor {
public:
  PacketProcessor(state::WorldState &worldState,
                  broker::PacketBroker &brokerSystem,
                  broker::EventBroker &eventBroker,
                  const pk2::GameData &gameData);

  void initialize();
  void handlePacket(const PacketContainer &packet) const;
private:
  state::WorldState &worldState_;
  broker::PacketBroker &packetBroker_;
  broker::EventBroker &eventBroker_;
  const pk2::GameData &gameData_;
  packet::parsing::PacketParser packetParser_{worldState_.entityTracker(), gameData_};

  void subscribeToPackets();
  void resetDataBecauseCharacterSpawned() const;

  // Packet handle functions
  //  In principal, each of these functions should only update the state and maybe publish an event.
  //  All member functions are const because this class should hold no state.
  // From LoginModule
  void serverListReceived(const packet::parsing::ParsedLoginServerList &packet) const;
  void loginResponseReceived(const packet::parsing::ParsedLoginResponse &packet) const;
  void loginClientInfoReceived(const packet::parsing::ParsedLoginClientInfo &packet) const;
  void unknownPacketReceived(const packet::parsing::ParsedUnknown &packet) const;
  void serverAuthReceived(const packet::parsing::ParsedServerAuthResponse &packet) const;
  void charListReceived(const packet::parsing::ParsedServerAgentCharacterSelectionActionResponse &packet) const;
  void charSelectionJoinResponseReceived(const packet::parsing::ParsedServerAgentCharacterSelectionJoinResponse &packet) const;
  // From MovementModule
  void serverAgentEntityUpdateAngleReceived(packet::parsing::ServerAgentEntityUpdateAngle &packet) const;
  void serverAgentEntityUpdateMovementReceived(packet::parsing::ServerAgentEntityUpdateMovement &packet) const;
  void serverAgentEntitySyncPositionReceived(packet::parsing::ServerAgentEntitySyncPosition &packet) const;
  void serverAgentEntityUpdatePositionReceived(packet::parsing::ServerAgentEntityUpdatePosition &packet) const;
  // From CharacterInfoModule
  void clientItemMoveReceived(const packet::parsing::ParsedClientItemMove &packet) const;
  void serverAgentCharacterDataReceived(const packet::parsing::ServerAgentCharacterData &packet) const;
  void serverAgentCosDataReceived(const packet::parsing::ServerAgentCosData &packet) const;
  void serverAgentInventoryStorageDataReceived(const packet::parsing::ParsedServerAgentInventoryStorageData &packet) const;
  void serverAgentEntityUpdateStateReceived(packet::parsing::ServerAgentEntityUpdateState &packet) const;
  void serverAgentEntityUpdateMoveSpeedReceived(const packet::parsing::ServerAgentEntityUpdateMoveSpeed &packet) const;
  void serverAgentEntityRemoveOwnershipReceived(const packet::parsing::ServerAgentEntityRemoveOwnership &packet) const;
  void serverAgentEntityUpdateStatusReceived(const packet::parsing::ServerAgentEntityUpdateStatus &packet) const;
  void serverAgentAbnormalInfoReceived(const packet::parsing::ParsedServerAgentAbnormalInfo &packet) const;
  void serverAgentCharacterUpdateStatsReceived(const packet::parsing::ParsedServerAgentCharacterUpdateStats &packet) const;
  void serverAgentInventoryItemUseResponseReceived(const packet::parsing::ServerAgentInventoryItemUseResponse &packet) const;
  void serverAgentInventoryOperationResponseReceived(const packet::parsing::ServerAgentInventoryOperationResponse &packet) const;
  void serverAgentEntityGroupSpawnDataReceived(const packet::parsing::ServerAgentEntityGroupSpawnData &packet) const;
  void serverAgentEntitySpawnReceived(const packet::parsing::ServerAgentEntitySpawn &packet) const;
  void serverAgentEntityDespawnReceived(const packet::parsing::ServerAgentEntityDespawn &packet) const;
  void entitySpawned(std::shared_ptr<entity::Entity> entity) const;
  void entityDespawned(sro::scalar_types::EntityGlobalId globalId) const;

  // Misc
  void serverAgentDeselectResponseReceived(const packet::parsing::ServerAgentActionDeselectResponse &packet) const;
  void serverAgentSelectResponseReceived(const packet::parsing::ServerAgentActionSelectResponse &packet) const;
  void serverAgentTalkResponseReceived(const packet::parsing::ServerAgentActionTalkResponse &packet) const;
  void serverAgentInventoryRepairResponseReceived(const packet::parsing::ServerAgentInventoryRepairResponse &packet) const;
  void serverAgentInventoryUpdateDurabilityReceived(const packet::parsing::ServerAgentInventoryUpdateDurability &packet) const;
  void serverAgentInventoryUpdateItemReceived(const packet::parsing::ServerAgentInventoryUpdateItem &packet) const;
  // void clientAgentActionDeselectRequestReceived(const packet::parsing::ClientAgentActionDeselectRequest &packet) const;
  // void clientAgentActionSelectRequestReceived(const packet::parsing::ClientAgentActionSelectRequest &packet) const;
  void clientAgentActionTalkRequestReceived(const packet::parsing::ClientAgentActionTalkRequest &packet) const;
  void serverAgentEntityUpdatePointsReceived(const packet::parsing::ServerAgentEntityUpdatePoints &packet) const;
  void serverAgentEntityUpdateExperienceReceived(const packet::parsing::ServerAgentEntityUpdateExperience &packet) const;
  void serverAgentGuildStorageDataReceived(const packet::parsing::ServerAgentGuildStorageData &packet) const;
  void clientAgentActionCommandRequestReceived(const packet::parsing::ClientAgentActionCommandRequest &packet) const;
  void serverAgentActionCommandResponseReceived(const packet::parsing::ServerAgentActionCommandResponse &packet) const;
  void serverAgentSkillBeginReceived(const packet::parsing::ServerAgentSkillBegin &packet) const;
  void serverAgentSkillEndReceived(const packet::parsing::ServerAgentSkillEnd &packet) const;
  void handleSkillAction(const packet::structures::SkillAction &action, std::optional<sro::scalar_types::EntityGlobalId> globalId = std::nullopt) const;
  void handleKnockedBackOrKnockedDown() const;
  void serverAgentBuffAddReceived(const packet::parsing::ServerAgentBuffAdd &packet) const;
  void serverAgentBuffRemoveReceived(const packet::parsing::ServerAgentBuffRemove &packet) const;

  std::optional<std::chrono::milliseconds> getItemCooldownMs(const storage::ItemExpendable &item) const;
  void printCommandQueues() const;
};

#endif // PACKETPROCESSOR_HPP_