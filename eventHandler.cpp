#include "eventHandler.hpp"

namespace {

sro::entity::LifeState lifeStateFromProto(const broadcast::LifeState lifeState) {
  switch (lifeState) {
    case broadcast::LifeState::kEmbryo:
      return sro::entity::LifeState::kEmbryo;
    case broadcast::LifeState::kAlive:
      return sro::entity::LifeState::kAlive;
    case broadcast::LifeState::kDead:
      return sro::entity::LifeState::kDead;
    case broadcast::LifeState::kGone:
      return sro::entity::LifeState::kGone;
    default:
      throw std::runtime_error("Unknown lifestate");
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

namespace {

sro::Position parsePosition(const broadcast::Position &pos) {
  return {static_cast<sro::RegionId>(pos.regionid()), pos.x(), pos.y(), pos.z()};
}

} // namespace

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
    case broadcast::BroadcastMessage::BodyCase::kCharacterPositionChanged: {
        const broadcast::CharacterPositionChanged &msg = message.characterpositionchanged();
        emit characterPositionChanged(parsePosition(msg.position()));
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kCharacterMovementBegan: {
        const broadcast::CharacterMovementBegan &msg = message.charactermovementbegan();
        const sro::Position currentPosition = parsePosition(msg.currentposition());
        const auto speed = msg.speed();
        switch (msg.destination_case()) {
          case broadcast::CharacterMovementBegan::DestinationCase::kDestinationPosition: {
            emit characterMovementBeganToDest(currentPosition, parsePosition(msg.destinationposition()), speed);
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
        emit characterMovementEnded(parsePosition(msg.currentposition()));
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kCharacterNotMovingAngleChanged: {
        const broadcast::CharacterNotMovingAngleChanged &msg = message.characternotmovinganglechanged();
        emit characterNotMovingAngleChanged(msg.angle());
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
        emit entitySpawned(msg.globalid(), parsePosition(msg.position()), msg.entitytype());
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kEntityDespawned: {
        const broadcast::EntityDespawned &msg = message.entitydespawned();
        emit entityDespawned(msg.globalid());
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kEntityPositionChanged: {
        const broadcast::EntityPositionChanged &msg = message.entitypositionchanged();
        emit entityPositionChanged(msg.globalid(), parsePosition(msg.position()));
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kEntityMovementBegan: {
        const broadcast::EntityMovementBegan &msg = message.entitymovementbegan();
        const auto entityId = msg.globalid();
        const broadcast::CharacterMovementBegan &charMovementMsg = msg.charactermovementbegan();
        sro::Position currentPosition = parsePosition(charMovementMsg.currentposition());
        const auto speed = charMovementMsg.speed();
        switch (charMovementMsg.destination_case()) {
          case broadcast::CharacterMovementBegan::DestinationCase::kDestinationPosition: {
            emit entityMovementBeganToDest(entityId, currentPosition, parsePosition(charMovementMsg.destinationposition()), speed);
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
        emit entityMovementEnded(entityId, parsePosition(charMovementMsg.currentposition()));
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kEntityLifeStateChanged: {
        const broadcast::EntityLifeStateChanged &msg = message.entitylifestatechanged();
        const auto entityId = msg.globalid();
        const auto lifeState = lifeStateFromProto(msg.lifestate());
        emit entityLifeStateChanged(entityId, lifeState);
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kTrainingAreaSet: {
        const broadcast::TrainingAreaSet &msg = message.trainingareaset();
        switch (msg.geometry_case()) {
          case broadcast::TrainingAreaSet::GeometryCase::kCircle: {
            emit trainingAreaCircleSet(parsePosition(msg.circle().center()), msg.circle().radius());
            break;
          }
          default:
            throw std::runtime_error("Unhandled training area geometry");
        }
        break;
      }
    case broadcast::BroadcastMessage::BodyCase::kTrainingAreaReset: {
        emit trainingAreaReset();
        break;
      }
    default:
      // Unknown case. Might be a malformed message
      break;
  }
}