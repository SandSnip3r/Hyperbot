#ifndef UI_USERINTERFACE_HPP_
#define UI_USERINTERFACE_HPP_

#include "broker/eventBroker.hpp"
#include "packet/structures/packetInnerStructures.hpp"

#include "ui-proto/broadcast.pb.h"

#include <zmq.hpp>

#include <string_view>

namespace ui {

class UserInterface {
// TODO: std::string vs std::string_view. Might need to compile protobuf with c++17?
public:
  UserInterface(broker::EventBroker &eventBroker);
  ~UserInterface();
  void run();
  void broadcastCharacterHpUpdate(uint32_t currentHp);
  void broadcastCharacterMpUpdate(uint32_t currentMp);
  void broadcastCharacterMaxHpMpUpdate(uint32_t maxHp, uint32_t maxMp);
  void broadcastCharacterLevelUpdate(uint8_t currentLevel, int64_t expRequired);
  void broadcastCharacterExperienceUpdate(uint64_t currentExperience, uint32_t currentSpExperience);
  void broadcastCharacterSpUpdate(uint32_t skillPoints);
  void broadcastCharacterNameUpdate(std::string_view characterName);
  void broadcastGoldAmountUpdate(uint64_t goldAmount, broadcast::GoldLocation goldLocation);
  void broadcastMovementBeganUpdate(const packet::structures::Position &srcPosition, const packet::structures::Position &destPosition, float speed);
  void broadcastMovementBeganUpdate(const packet::structures::Position &srcPosition, uint16_t angle, float speed);
  void broadcastMovementEndedUpdate(const packet::structures::Position &currentPosition);
  void broadcastRegionNameUpdate(std::string_view regionName);
  void broadcast(const broadcast::BroadcastMessage &broadcastProto);
private:
  zmq::context_t context_;
  zmq::socket_t publisher_{context_, zmq::socket_type::pub};
  broker::EventBroker &eventBroker_;
  std::thread thr_;
  void privateRun();
  void handle(const zmq::message_t &request);
};

} // namespace ui

#endif // UI_USERINTERFACE_HPP_