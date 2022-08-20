#ifndef UI_USERINTERFACE_HPP_
#define UI_USERINTERFACE_HPP_

#include "broker/eventBroker.hpp"

#include "ui-proto/broadcast.pb.h"

#include <zmq.hpp>

#include <string_view>

namespace ui {

class UserInterface {
public:
  UserInterface(broker::EventBroker &eventBroker);
  ~UserInterface();
  void run();
  void broadcastCharacterLevelUpdate(uint8_t currentLevel, int64_t expRequired);
  void broadcastCharacterExperienceUpdate(uint64_t currentExperience, uint32_t currentSpExperience);
  void broadcastCharacterSpUpdate(uint32_t skillPoints);
  void broadcastCharacterNameUpdate(std::string_view characterName);
  void broadcastGoldAmountUpdate(uint64_t goldAmount, broadcast::GoldLocation goldLocation);
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