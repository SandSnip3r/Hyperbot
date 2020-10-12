#ifndef MODULE_SKILL_USE_MODULE_HPP_
#define MODULE_SKILL_USE_MODULE_HPP_

#include "../broker/eventBroker.hpp"
#include "../broker/packetBroker.hpp"
#include "../packet/building/clientAgentActionCommandRequest.hpp"
#include "../packet/parsing/packetParser.hpp"
#include "../packet/parsing/clientAgentChatRequest.hpp"
#include "../packet/parsing/serverAgentActionSelectResponse.hpp"
#include "../pk2/gameData.hpp"
#include "../shared/silkroad_security.h"
#include "../state/entity.hpp"
#include "../state/self.hpp"
#include "../storage/storage.hpp"

#include "../packet/parsing/serverAgentSkillBegin.hpp"
#include "../packet/parsing/serverAgentSkillEnd.hpp"

#include <mutex>

namespace module {

class SkillUseModule {
public:
  SkillUseModule(state::Entity &entityState,
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

  // Packet handling functions
  bool clientAgentChatRequestReceived(packet::parsing::ClientAgentChatRequest &packet);
  void serverAgentActionSelectResponseReceived(packet::parsing::ServerAgentActionSelectResponse &packet);
  
  // General functions
  void selectEntity(state::Entity::EntityId entityId);
  void commonAttackEntity(state::Entity::EntityId entityId);
  void traceEntity(state::Entity::EntityId entityId);
  void pickupEntity(state::Entity::EntityId entityId);
  void serverAgentSkillBeginReceived(packet::parsing::ServerAgentSkillBegin &packet);
  void serverAgentSkillEndReceived(packet::parsing::ServerAgentSkillEnd &packet);
};

} // namespace module

#endif // MODULE_SKILL_USE_MODULE_HPP_