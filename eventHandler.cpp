#include "eventHandler.hpp"

EventHandler::EventHandler(zmq::context_t &context) : context_(context) {}

EventHandler::~EventHandler() {
  run_ = false;
  if (thr_.joinable()) {
    thr_.join();
  }
}

void EventHandler::runAsync() {
  run_ = true;
  thr_ = std::thread(&EventHandler::run, this);
}

void EventHandler::run() {
  // Set up subscriber socket
  zmq::socket_t subscriber(context_, zmq::socket_type::sub);
  subscriber.set(zmq::sockopt::subscribe, "");
  subscriber.connect("tcp://localhost:5556");
  emit connected();
  while (run_) {
    zmq::message_t message;
    subscriber.recv(message);
    broadcast::BroadcastMessage broadcastMessage;
    broadcastMessage.ParseFromArray(message.data(), message.size());
    handle(broadcastMessage);
  }
}

void EventHandler::handle(const broadcast::BroadcastMessage &message) {
  switch (message.body_case()) {
    case broadcast::BroadcastMessage::BodyCase::kCharacterHpUpdate: {
        const broadcast::CharacterHpUpdate &msg = message.characterhpupdate();
        emit characterHpUpdateChanged(msg.currenthp());
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kCharacterMpUpdate: {
        const broadcast::CharacterMpUpdate &msg = message.charactermpupdate();
        emit characterMpUpdateChanged(msg.currentmp());
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kCharacterMaxHpMpUpdate: {
        const broadcast::CharacterMaxHpMpUpdate &msg = message.charactermaxhpmpupdate();
        emit characterMaxHpMpUpdateChanged(msg.maxhp(), msg.maxmp());
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kCharacterLevelUpdate: {
        const broadcast::CharacterLevelUpdate &msg = message.characterlevelupdate();
        emit characterLevelUpdate(msg.level(), msg.exprequired());
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kCharacterExperienceUpdate: {
        const broadcast::CharacterExperienceUpdate &msg = message.characterexperienceupdate();
        emit characterExperienceUpdate(msg.currentexperience(), msg.currentspexperience());
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kCharacterSpUpdate: {
        const broadcast::CharacterSpUpdate &msg = message.characterspupdate();
        emit characterSpUpdate(msg.skillpoints());
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kCharacterNameUpdate: {
        const broadcast::CharacterNameUpdate &msg = message.characternameupdate();
        emit characterNameUpdate(msg.name());
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kGoldAmountUpdate: {
        const broadcast::GoldAmountUpdate &msg = message.goldamountupdate();
        if (msg.goldlocation() == broadcast::GoldLocation::kInventory) {
          emit inventoryGoldAmountUpdate(msg.goldamount());
        }
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kRegionNameUpdate: {
        const broadcast::RegionNameUpdate &msg = message.regionnameupdate();
        emit regionNameUpdate(msg.name());
        break;
      }
    default:
      // Unknown case. Might be a malformed message
      break;
  }
}
