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
    case broadcast::BroadcastMessage::BodyCase::kCharacterSpawn: {
        emit characterSpawn();
        break;
      }
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
        if (msg.goldlocation() == broadcast::ItemLocation::kCharacterInventory) {
          emit inventoryGoldAmountUpdate(msg.goldamount());
        } else if (msg.goldlocation() == broadcast::ItemLocation::kStorage) {
          emit storageGoldAmountUpdate(msg.goldamount());
        } else if (msg.goldlocation() == broadcast::ItemLocation::kGuildStorage) {
          emit guildStorageGoldAmountUpdate(msg.goldamount());
        }
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kCharacterMovementBegan: {
        const broadcast::CharacterMovementBegan &msg = message.charactermovementbegan();
        const broadcast::Position &currPos = msg.currentposition();
        sro::Position currentPosition(currPos.regionid(), currPos.x(), currPos.y(), currPos.z());
        const auto speed = msg.speed();
        switch (msg.destination_case()) {
          case broadcast::CharacterMovementBegan::DestinationCase::kDestinationPosition: {
            const broadcast::Position &destPos = msg.destinationposition();
            sro::Position destinationPosition(destPos.regionid(), destPos.x(), destPos.y(), destPos.z());
            emit characterMovementBeganToDest(currentPosition, destinationPosition, speed);
            break;
          }
          case broadcast::CharacterMovementBegan::DestinationCase::kDestinationAngle: {
            const auto angle = msg.destinationangle();
            emit characterMovementBeganTowardAngle(currentPosition, angle, speed);
            break;
          }
          default:
            // Unknown case. Might be a malformed message
            break;
        }
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kCharacterMovementEnded: {
        const broadcast::CharacterMovementEnded &msg = message.charactermovementended();
        const broadcast::Position &currPos = msg.currentposition();
        sro::Position currentPosition(currPos.regionid(), currPos.x(), currPos.y(), currPos.z());
        emit characterMovementEnded(currentPosition);
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kRegionNameUpdate: {
        const broadcast::RegionNameUpdate &msg = message.regionnameupdate();
        emit regionNameUpdate(msg.name());
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kItemUpdate: {
        const broadcast::ItemUpdate &msg = message.itemupdate();
        std::optional<std::string> itemName;
        if (msg.has_itemname()) {
          itemName = msg.itemname();
        }
        if (msg.itemlocation() == broadcast::ItemLocation::kCharacterInventory) {
          emit characterInventoryItemUpdate(msg.slotindex(), msg.quantity(), itemName);
        } else if (msg.itemlocation() == broadcast::ItemLocation::kAvatarInventory) {
          emit avatarInventoryItemUpdate(msg.slotindex(), msg.quantity(), itemName);
        } else if (msg.itemlocation() == broadcast::ItemLocation::kCosInventory) {
          emit cosInventoryItemUpdate(msg.slotindex(), msg.quantity(), itemName);
        } else if (msg.itemlocation() == broadcast::ItemLocation::kStorage) {
          emit storageItemUpdate(msg.slotindex(), msg.quantity(), itemName);
        } else if (msg.itemlocation() == broadcast::ItemLocation::kGuildStorage) {
          emit guildStorageItemUpdate(msg.slotindex(), msg.quantity(), itemName);
        }
        break;
      }
    default:
      // Unknown case. Might be a malformed message
      break;
  }
}