#ifndef PACKETPROCESSOR_HPP_
#define PACKETPROCESSOR_HPP_

#include "broker/eventBroker.hpp"
#include "broker/packetBroker.hpp"
#include "packet/parsing/serverAgentCharacterData.hpp"
#include "packet/parsing/serverAgentEntitySyncPosition.hpp"
#include "packet/parsing/serverAgentEntityUpdateMovement.hpp"
#include "packet/parsing/serverAgentEntityUpdateMoveSpeed.hpp"
#include "packet/parsing/serverAgentEntityUpdatePosition.hpp"
#include "packet/parsing/serverAgentEntityUpdateState.hpp"
#include "packet/parsing/packetParser.hpp"
#include "pk2/gameData.hpp"
#include "state/entity.hpp"
#include "state/self.hpp"
#include "ui/userInterface.hpp"

/*  PacketProcessor
 *  As packets come in, this class will update the state
 */
class PacketProcessor {
public:
  PacketProcessor(state::Entity &entityState,
                  state::Self &selfState,
                  broker::PacketBroker &brokerSystem,
                  broker::EventBroker &eventBroker,
                  ui::UserInterface &userInterface,
                  const packet::parsing::PacketParser &packetParser,
                  const pk2::GameData &gameData);

  bool handlePacket(const PacketContainer &packet) const;
private:
  state::Entity &entityState_;
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
  bool serverAgentEntityUpdateMovementReceived(packet::parsing::ServerAgentEntityUpdateMovement &packet) const;
  bool serverAgentEntitySyncPositionReceived(packet::parsing::ServerAgentEntitySyncPosition &packet) const;
  bool serverAgentEntityUpdatePositionReceived(packet::parsing::ServerAgentEntityUpdatePosition &packet) const;
  // From CharacterInfoModule
  bool clientItemMoveReceived(const packet::parsing::ParsedClientItemMove &packet) const;
  bool serverAgentCharacterDataReceived(const packet::parsing::ParsedServerAgentCharacterData &packet) const;
  bool serverAgentEntityUpdateStateReceived(packet::parsing::ServerAgentEntityUpdateState &packet) const;
  bool serverAgentEntityUpdateMoveSpeedReceived(const packet::parsing::ServerAgentEntityUpdateMoveSpeed &packet) const;
  bool serverAgentEntityUpdateStatusReceived(const packet::parsing::ParsedServerAgentEntityUpdateStatus &packet) const;
  bool serverAgentAbnormalInfoReceived(const packet::parsing::ParsedServerAgentAbnormalInfo &packet) const;
  bool serverAgentCharacterUpdateStatsReceived(const packet::parsing::ParsedServerAgentCharacterUpdateStats &packet) const;
  bool serverAgentInventoryItemUseResponseReceived(const packet::parsing::ParsedServerAgentInventoryItemUseResponse &packet) const;
  bool serverAgentInventoryOperationResponseReceived(const packet::parsing::ParsedServerAgentInventoryOperationResponse &packet) const;
  bool serverAgentEntityGroupSpawnDataReceived(const packet::parsing::ParsedServerAgentEntityGroupSpawnData &packet) const;
  bool serverAgentSpawnReceived(const packet::parsing::ParsedServerAgentSpawn &packet) const;
  bool serverAgentDespawnReceived(const packet::parsing::ParsedServerAgentDespawn &packet) const;
};

#endif // PACKETPROCESSOR_HPP_