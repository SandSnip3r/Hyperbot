#include "skillUseModule.hpp"
#include "../packet/opcode.hpp"
#include "../packet/building/clientAgentActionSelectRequest.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <regex>

namespace module {

SkillUseModule::SkillUseModule(state::Entity &entityState,
                               state::Self &selfState,
                               storage::Storage &inventory,
                               broker::PacketBroker &brokerSystem,
                               broker::EventBroker &eventBroker,
                               const packet::parsing::PacketParser &packetParser,
                               const pk2::GameData &gameData) :
      entityState_(entityState),
      selfState_(selfState),
      inventory_(inventory),
      broker_(brokerSystem),
      eventBroker_(eventBroker),
      packetParser_(packetParser),
      gameData_(gameData) {
  auto packetHandleFunction = std::bind(&SkillUseModule::handlePacket, this, std::placeholders::_1);
  // Client packets
  broker_.subscribeToClientPacket(packet::Opcode::kClientAgentChatRequest, packetHandleFunction);
  // Server packets
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentActionSelectResponse, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentSkillBegin, packetHandleFunction);
  broker_.subscribeToServerPacket(packet::Opcode::kServerAgentSkillEnd, packetHandleFunction);
  


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

  auto *clientChat = dynamic_cast<packet::parsing::ClientAgentChatRequest*>(parsedPacket.get());
  if (clientChat != nullptr) {
    return clientAgentChatRequestReceived(*clientChat);
  }

  auto *actionSelectResponse = dynamic_cast<packet::parsing::ServerAgentActionSelectResponse*>(parsedPacket.get());
  if (actionSelectResponse != nullptr) {
    serverAgentActionSelectResponseReceived(*actionSelectResponse);
    return true;
  }

  auto *serverAgentSkillBegin = dynamic_cast<packet::parsing::ServerAgentSkillBegin*>(parsedPacket.get());
  if (serverAgentSkillBegin != nullptr) {
    serverAgentSkillBeginReceived(*serverAgentSkillBegin);
    return true;
  }

  auto *serverAgentSkillEnd = dynamic_cast<packet::parsing::ServerAgentSkillEnd*>(parsedPacket.get());
  if (serverAgentSkillEnd != nullptr) {
    serverAgentSkillEndReceived(*serverAgentSkillEnd);
    return true;
  }

  std::cout << "SkillUseModule: Unhandled packet subscribed to\n";
  return true;
}

void SkillUseModule::serverAgentSkillBeginReceived(packet::parsing::ServerAgentSkillBegin &packet) {
  // std::ofstream damageFile("dmg.txt", std::ios_base::app);
  // if (!damageFile) {
  //   std::cout << "Cannot open file!!\n";
  //   return;
  // }
  // if (packet.result() && packet.casterGlobalId() == selfState_.globalId()) {
  //   for (const auto &hitObject : packet.action().hitObjects) {
  //     for (const auto &hit : hitObject.hits) {
  //       if (hit.damageFlag == packet::enums::DamageFlag::kNormal) {
  //         damageFile << "normal," << hit.damage << '\n';
  //       } else if (hit.damageFlag == packet::enums::DamageFlag::kCritical) {
  //         damageFile << "critical," << hit.damage << '\n';
  //       }
  //     }
  //   }
  // }
}

void SkillUseModule::serverAgentSkillEndReceived(packet::parsing::ServerAgentSkillEnd &packet) {
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

void SkillUseModule::commonAttackEntity(state::Entity::EntityId entityId) {
  std::cout << "Being asked to common attack " << entityId << '\n';
  if (entityState_.trackingEntity(entityId)) {
    auto entity = entityState_.getEntity(entityId);
    std::cout << "Common attacking ";
    packet::parsing::printObj(entity, gameData_);
    broker_.injectPacket(packet::building::ClientAgentActionCommandRequest::attack(entityId), PacketContainer::Direction::kClientToServer);
  }
}

void SkillUseModule::traceEntity(state::Entity::EntityId entityId) {
  std::cout << "Being asked to trace " << entityId << '\n';
  if (entityState_.trackingEntity(entityId)) {
    auto entity = entityState_.getEntity(entityId);
    std::cout << "Tracing ";
    packet::parsing::printObj(entity, gameData_);
    broker_.injectPacket(packet::building::ClientAgentActionCommandRequest::trace(entityId), PacketContainer::Direction::kClientToServer);
  }
}

void SkillUseModule::pickupEntity(state::Entity::EntityId entityId) {
  std::cout << "Being asked to pickup " << entityId << '\n';
  if (entityState_.trackingEntity(entityId)) {
    auto entity = entityState_.getEntity(entityId);
    std::cout << "Picking up ";
    packet::parsing::printObj(entity, gameData_);
    broker_.injectPacket(packet::building::ClientAgentActionCommandRequest::pickup(entityId), PacketContainer::Direction::kClientToServer);
  }
}

bool SkillUseModule::clientAgentChatRequestReceived(packet::parsing::ClientAgentChatRequest &packet) {
  std::regex selectGidRegex(R"delim((attack|trace|pickup|select) ([0-9]+))delim");
  std::smatch regexMatch;
  if (std::regex_match(packet.message(), regexMatch, selectGidRegex)) {
    const std::string operation = regexMatch[1].str();
    state::Entity::EntityId entityId = std::stoi(regexMatch[2].str());
    if (operation == "select") {
      selectEntity(entityId);
    } else if (operation == "attack") {
      commonAttackEntity(entityId);
    } else if (operation == "trace") {
      traceEntity(entityId);
    } else if (operation == "pickup") {
      pickupEntity(entityId);
    }
    return false;
  } else {
    return true;
  }
}

void SkillUseModule::serverAgentActionSelectResponseReceived(packet::parsing::ServerAgentActionSelectResponse &packet) {
  std::cout << "serverAgentActionSelectResponseReceived\n";
  if (packet.result() == 1) {
    // Successfully selected
    std::cout << "Selected successfully\n";
  } else {
    std::cout << "Selection unsuccessful! Error code " << (int)packet.errorCode() << '\n';
  }
}

} // namespace module