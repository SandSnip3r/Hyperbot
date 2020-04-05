#ifndef MODULE_SKILL_USE_MODULE_HPP_
#define MODULE_SKILL_USE_MODULE_HPP_

#include "../broker/eventBroker.hpp"
#include "../broker/packetBroker.hpp"
#include "../packet/parsing/packetParser.hpp"
// #include "../packet/parsing/parsedPacket.hpp"
#include "../packet/parsing/clientAgentChatRequest.hpp"
#include "../pk2/gameData.hpp"
#include "../shared/silkroad_security.h"
#include "../state/entity.hpp"

#include <mutex>

namespace module {

class SkillUseModule {
public:
  SkillUseModule(state::Entity &entityState,
                 broker::PacketBroker &brokerSystem,
                 broker::EventBroker &eventBroker,
                 const packet::parsing::PacketParser &packetParser,
                 const pk2::GameData &gameData);
  bool handlePacket(const PacketContainer &packet);
private:
  state::Entity &entityState_;
  broker::PacketBroker &broker_;
  broker::EventBroker &eventBroker_;
  const packet::parsing::PacketParser &packetParser_;
  const pk2::GameData &gameData_;
  std::mutex contentionProtectionMutex_;

  // Packet handling functions
  bool clientAgentChatRequestReceived(packet::parsing::ParsedClientAgentChatRequest &packet);
  
  // General functions
  void selectEntity(state::Entity::EntityId entityId);
};

} // namespace module

#endif // MODULE_SKILL_USE_MODULE_HPP_