#include "eventHandler.hpp"

namespace {

sro::types::EntityType entityTypeFromProtoEntityType(broadcast::EntityType type) {
  switch (type) {
    case broadcast::EntityType::kSelf:
      return sro::types::EntityType::kSelf;
      break;
    case broadcast::EntityType::kCharacter:
      return sro::types::EntityType::kCharacter;
      break;
    case broadcast::EntityType::kPlayerCharacter:
      return sro::types::EntityType::kPlayerCharacter;
      break;
    case broadcast::EntityType::kNonplayerCharacter:
      return sro::types::EntityType::kNonplayerCharacter;
      break;
    case broadcast::EntityType::kMonster:
      return sro::types::EntityType::kMonster;
      break;
    case broadcast::EntityType::kItem:
      return sro::types::EntityType::kItem;
      break;
    case broadcast::EntityType::kPortal:
      return sro::types::EntityType::kPortal;
      break;
    default:
      throw std::runtime_error("Unknown entity type from proto");
  }
}

} // anonymous namespace

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
    case broadcast::BroadcastMessage::BodyCase::kEntitySpawned: {
        const broadcast::EntitySpawned &msg = message.entityspawned();
        const broadcast::Position &pos = msg.position();
        sro::Position sroPos(pos.regionid(), pos.x(), pos.y(), pos.z());
        sro::types::EntityType entityType = entityTypeFromProtoEntityType(msg.entitytype());
        emit entitySpawned(msg.globalid(), sroPos, entityType);
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kEntityDespawned: {
        const broadcast::EntityDespawned &msg = message.entitydespawned();
        emit entityDespawned(msg.globalid());
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kEntityPositionChanged: {
        const broadcast::EntityPositionChanged &msg = message.entitypositionchanged();
        const broadcast::Position &currPos = msg.position();
        sro::Position currentPosition(currPos.regionid(), currPos.x(), currPos.y(), currPos.z());
        emit entityPositionChanged(msg.globalid(), currentPosition);
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kEntityMovementBegan: {
        const broadcast::EntityMovementBegan &msg = message.entitymovementbegan();
        const auto entityId = msg.globalid();
        const broadcast::CharacterMovementBegan &charMovementMsg = msg.charactermovementbegan();
        const broadcast::Position &currPos = charMovementMsg.currentposition();
        sro::Position currentPosition(currPos.regionid(), currPos.x(), currPos.y(), currPos.z());
        const auto speed = charMovementMsg.speed();
        switch (charMovementMsg.destination_case()) {
          case broadcast::CharacterMovementBegan::DestinationCase::kDestinationPosition: {
            const broadcast::Position &destPos = charMovementMsg.destinationposition();
            sro::Position destinationPosition(destPos.regionid(), destPos.x(), destPos.y(), destPos.z());
            emit entityMovementBeganToDest(entityId, currentPosition, destinationPosition, speed);
            break;
          }
          case broadcast::CharacterMovementBegan::DestinationCase::kDestinationAngle: {
            const auto angle = charMovementMsg.destinationangle();
            emit entityMovementBeganTowardAngle(entityId, currentPosition, angle, speed);
            break;
          }
          default:
            // Unknown case. Might be a malformed message
            break;
        }
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kEntityMovementEnded: {
        const broadcast::EntityMovementEnded &msg = message.entitymovementended();
        const broadcast::CharacterMovementEnded &charMovementMsg = msg.charactermovementended();
        const auto entityId = msg.globalid();
        const broadcast::Position &currPos = charMovementMsg.currentposition();
        sro::Position currentPosition(currPos.regionid(), currPos.x(), currPos.y(), currPos.z());
        emit entityMovementEnded(entityId, currentPosition);
        break;
      }
    default:
      // Unknown case. Might be a malformed message
      break;
  }
}