#ifndef MODULE_MOVEMENT_MODULE_HPP_
#define MODULE_MOVEMENT_MODULE_HPP_

#include "../broker/eventBroker.hpp"
#include "../broker/packetBroker.hpp"
#include "../math/vector.hpp"
#include "../packet/parsing/clientAgentCharacterMoveRequest.hpp"
#include "../packet/parsing/clientAgentChatRequest.hpp"
#include "../packet/parsing/packetParser.hpp"
#include "../packet/parsing/serverAgentEntitySyncPosition.hpp"
#include "../packet/parsing/serverAgentEntityUpdateMovement.hpp"
#include "../packet/parsing/serverAgentEntityUpdatePosition.hpp"
#include "../pk2/gameData.hpp"
#include "../shared/silkroad_security.h"
#include "../state/entity.hpp"
#include "../state/self.hpp"
#include "../storage/storage.hpp"

#include "pathfinder.h"

#include <list>
#include <mutex>
#include <random>

namespace module {

class MovementModule {
public:
  MovementModule(state::Entity &entityState,
                 state::Self &selfState,
                 storage::Storage &inventory,
                 broker::PacketBroker &brokerSystem,
                 broker::EventBroker &eventBroker,
                 const packet::parsing::PacketParser &packetParser,
                 const pk2::GameData &gameData);
  bool handlePacket(const PacketContainer &packet);
private:
  state::Entity &entityState_;
  state::Self &selfState_;
  storage::Storage &inventory_;
  broker::PacketBroker &broker_;
  broker::EventBroker &eventBroker_;
  const packet::parsing::PacketParser &packetParser_;
  const pk2::GameData &gameData_;
  std::mutex contentionProtectionMutex_;

  float secondsToTravel(const packet::structures::Position &srcPosition, const packet::structures::Position &destPosition) const;

  // Packet handling functions
  bool clientAgentChatRequestReceived(packet::parsing::ClientAgentChatRequest &packet);
  bool clientAgentCharacterMoveRequestReceived(packet::parsing::ClientAgentCharacterMoveRequest &packet);
  bool serverAgentEntityUpdateMovementReceived(packet::parsing::ServerAgentEntityUpdateMovement &packet);
  bool serverAgentEntitySyncPositionReceived(packet::parsing::ServerAgentEntitySyncPosition &packet);
  bool serverAgentEntityUpdatePositionReceived(packet::parsing::ServerAgentEntityUpdatePosition &packet);

  // Event handling functions
  void handleEvent(const event::Event *event);
  void handleMovementEnded();
  void handleSpeedUpdated();

  // General functions
};

} // namespace module

#endif // MODULE_MOVEMENT_MODULE_HPP_