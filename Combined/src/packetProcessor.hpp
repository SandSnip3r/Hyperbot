#ifndef PACKETPROCESSOR_HPP_
#define PACKETPROCESSOR_HPP_

#include "broker/eventBroker.hpp"
#include "broker/packetBroker.hpp"
// #include "packet/parsing/clientAgentActionDeselectRequest.hpp"
// #include "packet/parsing/clientAgentActionSelectRequest.hpp"
#include "packet/parsing/clientAgentActionTalkRequest.hpp"
#include "packet/parsing/serverAgentActionCommandResponse.hpp"
#include "packet/parsing/serverAgentActionDeselectResponse.hpp"
#include "packet/parsing/serverAgentActionSelectResponse.hpp"
#include "packet/parsing/serverAgentActionTalkResponse.hpp"
#include "packet/parsing/serverAgentCharacterData.hpp"
#include "packet/parsing/serverAgentCosData.hpp"
#include "packet/parsing/serverAgentEntitySyncPosition.hpp"
#include "packet/parsing/serverAgentEntityUpdateAngle.hpp"
#include "packet/parsing/serverAgentEntityUpdateExperience.hpp"
#include "packet/parsing/serverAgentEntityUpdateMovement.hpp"
#include "packet/parsing/serverAgentEntityUpdateMoveSpeed.hpp"
#include "packet/parsing/serverAgentEntityUpdatePoints.hpp"
#include "packet/parsing/serverAgentEntityUpdatePosition.hpp"
#include "packet/parsing/serverAgentEntityUpdateState.hpp"
#include "packet/parsing/serverAgentGuildStorageData.hpp"
#include "packet/parsing/serverAgentInventoryOperationResponse.hpp"
#include "packet/parsing/serverAgentInventoryRepairResponse.hpp"
#include "packet/parsing/serverAgentInventoryStorageData.hpp"
#include "packet/parsing/serverAgentInventoryUpdateDurability.hpp"
#include "packet/parsing/serverAgentInventoryUpdateItem.hpp"
#include "packet/parsing/serverAgentSkillBegin.hpp"
#include "packet/parsing/serverAgentSkillEnd.hpp"
#include "packet/parsing/packetParser.hpp"
#include "pk2/gameData.hpp"
#include "state/entityTracker.hpp"
#include "state/self.hpp"
#include "ui/userInterface.hpp"

#define ENFORCE_PURIFICATION_PILL_COOLDOWN

/*  PacketProcessor
 *  As packets come in, this class will update the state
 */
class PacketProcessor {
public:
  PacketProcessor(state::EntityTracker &entityTracker,
                  state::Self &selfState,
                  broker::PacketBroker &brokerSystem,
                  broker::EventBroker &eventBroker,
                  ui::UserInterface &userInterface,
                  const packet::parsing::PacketParser &packetParser,
                  const pk2::GameData &gameData);

  bool handlePacket(const PacketContainer &packet) const;
private:
  state::EntityTracker &entityTracker_;
  state::Self &selfState_;
  broker::PacketBroker &broker_;
  broker::EventBroker &eventBroker_;
  ui::UserInterface &userInterface_;
  const packet::parsing::PacketParser &packetParser_;
  const pk2::GameData &gameData_;

  // TODO: We should move this to a more global configuration area for general bot mechanics configuration
  //       Maybe we could try to improve this value based on item use results
  static const int kPotionDelayBufferMs_ = 225; //200 too fast sometimes, 300 seems always good

  void subscribeToPackets();
  void resetDataBecauseCharacterSpawned() const;

  // Packet handle functions
  //  In principal, each of these functions should only update the state and maybe publish an event.
  //  All member functions are const because this class should hold no state.
  // From LoginModule
  bool serverListReceived(const packet::parsing::ParsedLoginServerList &packet) const;
  bool loginResponseReceived(const packet::parsing::ParsedLoginResponse &packet) const;
  bool loginClientInfoReceived(const packet::parsing::ParsedLoginClientInfo &packet) const;
  bool unknownPacketReceived(const packet::parsing::ParsedUnknown &packet) const;
  bool serverAuthReceived(const packet::parsing::ParsedServerAuthResponse &packet) const;
  bool charListReceived(const packet::parsing::ParsedServerAgentCharacterSelectionActionResponse &packet) const;
  bool charSelectionJoinResponseReceived(const packet::parsing::ParsedServerAgentCharacterSelectionJoinResponse &packet) const;
  // From MovementModule
  bool serverAgentEntityUpdateAngleReceived(packet::parsing::ServerAgentEntityUpdateAngle &packet) const;
  bool serverAgentEntityUpdateMovementReceived(packet::parsing::ServerAgentEntityUpdateMovement &packet) const;
  bool serverAgentEntitySyncPositionReceived(packet::parsing::ServerAgentEntitySyncPosition &packet) const;
  bool serverAgentEntityUpdatePositionReceived(packet::parsing::ServerAgentEntityUpdatePosition &packet) const;
  // From CharacterInfoModule
  bool clientItemMoveReceived(const packet::parsing::ParsedClientItemMove &packet) const;
  bool serverAgentCharacterDataReceived(const packet::parsing::ParsedServerAgentCharacterData &packet) const;
  bool serverAgentCosDataReceived(const packet::parsing::ServerAgentCosData &packet) const;
  bool serverAgentInventoryStorageDataReceived(const packet::parsing::ParsedServerAgentInventoryStorageData &packet) const;
  bool serverAgentEntityUpdateStateReceived(packet::parsing::ServerAgentEntityUpdateState &packet) const;
  bool serverAgentEntityUpdateMoveSpeedReceived(const packet::parsing::ServerAgentEntityUpdateMoveSpeed &packet) const;
  bool serverAgentEntityUpdateStatusReceived(const packet::parsing::ParsedServerAgentEntityUpdateStatus &packet) const;
  bool serverAgentAbnormalInfoReceived(const packet::parsing::ParsedServerAgentAbnormalInfo &packet) const;
  bool serverAgentCharacterUpdateStatsReceived(const packet::parsing::ParsedServerAgentCharacterUpdateStats &packet) const;
  bool serverAgentInventoryItemUseResponseReceived(const packet::parsing::ParsedServerAgentInventoryItemUseResponse &packet) const;
  bool serverAgentInventoryOperationResponseReceived(const packet::parsing::ServerAgentInventoryOperationResponse &packet) const;
  bool serverAgentEntityGroupSpawnDataReceived(const packet::parsing::ParsedServerAgentEntityGroupSpawnData &packet) const;
  bool serverAgentSpawnReceived(const packet::parsing::ParsedServerAgentSpawn &packet) const;
  bool serverAgentDespawnReceived(const packet::parsing::ParsedServerAgentDespawn &packet) const;
  void entitySpawned(std::shared_ptr<entity::Entity> entity) const;
  void entityDespawned(sro::scalar_types::EntityGlobalId globalId) const;

  // Misc
  bool serverAgentDeselectResponseReceived(const packet::parsing::ServerAgentActionDeselectResponse &packet) const;
  bool serverAgentSelectResponseReceived(const packet::parsing::ServerAgentActionSelectResponse &packet) const;
  bool serverAgentTalkResponseReceived(const packet::parsing::ServerAgentActionTalkResponse &packet) const;
  bool serverAgentInventoryRepairResponseReceived(const packet::parsing::ServerAgentInventoryRepairResponse &packet) const;
  bool serverAgentInventoryUpdateDurabilityReceived(const packet::parsing::ServerAgentInventoryUpdateDurability &packet) const;
  bool serverAgentInventoryUpdateItemReceived(const packet::parsing::ServerAgentInventoryUpdateItem &packet) const;
  // bool clientAgentActionDeselectRequestReceived(const packet::parsing::ClientAgentActionDeselectRequest &packet) const;
  // bool clientAgentActionSelectRequestReceived(const packet::parsing::ClientAgentActionSelectRequest &packet) const;
  bool clientAgentActionTalkRequestReceived(const packet::parsing::ClientAgentActionTalkRequest &packet) const;
  bool serverAgentEntityUpdatePointsReceived(const packet::parsing::ServerAgentEntityUpdatePoints &packet) const;
  bool serverAgentEntityUpdateExperienceReceived(const packet::parsing::ServerAgentEntityUpdateExperience &packet) const;
  bool serverAgentGuildStorageDataReceived(const packet::parsing::ServerAgentGuildStorageData &packet) const;

  bool serverAgentActionCommandResponseReceived(const packet::parsing::ServerAgentActionCommandResponse &packet) const;
  bool serverAgentSkillBeginReceived(const packet::parsing::ServerAgentSkillBegin &packet) const;
  bool serverAgentSkillEndReceived(const packet::parsing::ServerAgentSkillEnd &packet) const;

  // Helpers
  entity::MobileEntity& getMobileEntity(sro::scalar_types::EntityGlobalId globalId) const;
};

#endif // PACKETPROCESSOR_HPP_