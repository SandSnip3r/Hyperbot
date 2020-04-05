#include "skillUseModule.hpp"
#include "../packet/opcode.hpp"
#include "../packet/building/clientAgentActionSelectRequest.hpp"

#include <iostream>
#include <memory>
#include <regex>

namespace module {

SkillUseModule::SkillUseModule(state::Entity &entityState,
                               broker::PacketBroker &brokerSystem,
                               broker::EventBroker &eventBroker,
                               const packet::parsing::PacketParser &packetParser,
                               const pk2::GameData &gameData) :
      entityState_(entityState),
      broker_(brokerSystem),
      eventBroker_(eventBroker),
      packetParser_(packetParser),
      gameData_(gameData) {
  auto packetHandleFunction = std::bind(&SkillUseModule::handlePacket, this, std::placeholders::_1);
  // Client packets
  broker_.subscribeToClientPacket(packet::Opcode::kClientAgentChatRequest, packetHandleFunction);
  // Server packets

  // TODO: Save subscription ID to possibly unsubscribe in the future
  // eventBroker_.subscribeToEvent(event::EventCode::kHpPotionCooldownEnded, std::bind(&SkillUseModule::handlePotionCooldownEnded, this, std::placeholders::_1));
}

// void SkillUseModule::handlePotionCooldownEnded(const std::unique_ptr<event::Event> &event) {
//   std::unique_lock<std::mutex> contentionProtectionLock(contentionProtectionMutex_);
//   auto eventCode = event->getEventCode();
//   if (eventCode == event::EventCode::kHpPotionCooldownEnded) {
//     hpPotionEventId_.reset();
//     checkIfNeedToHeal();
//   } else if (eventCode == event::EventCode::kMpPotionCooldownEnded) {
//     mpPotionEventId_.reset();
//     checkIfNeedToHeal();
//   } else if (eventCode == event::EventCode::kVigorPotionCooldownEnded) {
//     vigorPotionEventId_.reset();
//     checkIfNeedToHeal();
//   }
// }

bool SkillUseModule::handlePacket(const PacketContainer &packet) {
  std::unique_lock<std::mutex> contentionProtectionLock(contentionProtectionMutex_);
  std::unique_ptr<packet::parsing::ParsedPacket> parsedPacket;
  try {
    parsedPacket = packetParser_.parsePacket(packet);
  } catch (std::exception &ex) {
    std::cerr << "[SkillUseModule] Failed to parse packet " << std::hex << packet.opcode << std::dec << "\n  Error: \"" << ex.what() << "\"\n";
    return true;
  }

  if (!parsedPacket) {
    // Not yet parsing this packet
    return true;
  }

  auto *clientChat = dynamic_cast<packet::parsing::ParsedClientAgentChatRequest*>(parsedPacket.get());
  if (clientChat != nullptr) {
    return clientAgentChatRequestReceived(*clientChat);
  }

  std::cout << "Unhandled packet subscribed to\n";
  return true;
}

void SkillUseModule::selectEntity(state::Entity::EntityId entityId) {
  std::cout << "Being asked to select " << entityId << '\n';
  if (entityState_.trackingEntity(entityId)) {
    auto entity = entityState_.getEntity(entityId);
    std::cout << "Selecting ";
    packet::parsing::printObj(entity, gameData_);
    broker_.injectPacket(packet::building::ClientAgentActionSelectRequest::packet(entityId), PacketContainer::Direction::kClientToServer);
  }
}

bool SkillUseModule::clientAgentChatRequestReceived(packet::parsing::ParsedClientAgentChatRequest &packet) {
  std::regex selectGidRegex(R"delim(select ([0-9]+))delim");
  std::smatch regexMatch;
  if (std::regex_match(packet.message(), regexMatch, selectGidRegex)) {
    state::Entity::EntityId entityId = std::stoi(regexMatch[1].str());
    selectEntity(entityId);
    return false;
  } else {
    return true;
  }
}

} // namespace module