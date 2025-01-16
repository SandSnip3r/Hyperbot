#include "eventHandler.hpp"

#include "ui_proto/position.pb.h"

namespace {

sro::entity::LifeState lifeStateFromProto(const proto::broadcast::LifeState lifeState) {
  switch (lifeState) {
    case proto::broadcast::LifeState::kEmbryo:
      return sro::entity::LifeState::kEmbryo;
    case proto::broadcast::LifeState::kAlive:
      return sro::entity::LifeState::kAlive;
    case proto::broadcast::LifeState::kDead:
      return sro::entity::LifeState::kDead;
    case proto::broadcast::LifeState::kGone:
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
    proto::broadcast::BroadcastMessage broadcastMessage;
    broadcastMessage.ParseFromArray(message.data(), message.size());
    handle(broadcastMessage);
  }
}

namespace {

sro::Position parsePosition(const proto::position::Position &pos) {
  return {static_cast<sro::RegionId>(pos.regionid()), pos.x(), pos.y(), pos.z()};
}

} // namespace

void EventHandler::handle(const proto::broadcast::BroadcastMessage &message) {
  switch (message.body_case()) {
    case proto::broadcast::BroadcastMessage::BodyCase::kLaunch: {
        emit launch();
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kCharacterSpawn: {
        emit characterSpawn();
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kCharacterHpUpdate: {
        const proto::broadcast::CharacterHpUpdate &msg = message.characterhpupdate();
        emit characterHpUpdateChanged(msg.currenthp());
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kCharacterMpUpdate: {
        const proto::broadcast::CharacterMpUpdate &msg = message.charactermpupdate();
        emit characterMpUpdateChanged(msg.currentmp());
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kCharacterMaxHpMpUpdate: {
        const proto::broadcast::CharacterMaxHpMpUpdate &msg = message.charactermaxhpmpupdate();
        emit characterMaxHpMpUpdateChanged(msg.maxhp(), msg.maxmp());
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kCharacterLevelUpdate: {
        const proto::broadcast::CharacterLevelUpdate &msg = message.characterlevelupdate();
        emit characterLevelUpdate(msg.level(), msg.exprequired());
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kCharacterExperienceUpdate: {
        const proto::broadcast::CharacterExperienceUpdate &msg = message.characterexperienceupdate();
        emit characterExperienceUpdate(msg.currentexperience(), msg.currentspexperience());
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kCharacterSpUpdate: {
        const proto::broadcast::CharacterSpUpdate &msg = message.characterspupdate();
        emit characterSpUpdate(msg.skillpoints());
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kCharacterNameUpdate: {
        const proto::broadcast::CharacterNameUpdate &msg = message.characternameupdate();
        emit characterNameUpdate(msg.name());
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kGoldAmountUpdate: {
        const proto::broadcast::GoldAmountUpdate &msg = message.goldamountupdate();
        if (msg.goldlocation() == proto::broadcast::ItemLocation::kCharacterInventory) {
          emit inventoryGoldAmountUpdate(msg.goldamount());
        } else if (msg.goldlocation() == proto::broadcast::ItemLocation::kStorage) {
          emit storageGoldAmountUpdate(msg.goldamount());
        } else if (msg.goldlocation() == proto::broadcast::ItemLocation::kGuildStorage) {
          emit guildStorageGoldAmountUpdate(msg.goldamount());
        }
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kCharacterPositionChanged: {
        const proto::broadcast::CharacterPositionChanged &msg = message.characterpositionchanged();
        emit characterPositionChanged(parsePosition(msg.position()));
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kCharacterMovementBegan: {
        const proto::broadcast::CharacterMovementBegan &msg = message.charactermovementbegan();
        const sro::Position currentPosition = parsePosition(msg.currentposition());
        const auto speed = msg.speed();
        switch (msg.destination_case()) {
          case proto::broadcast::CharacterMovementBegan::DestinationCase::kDestinationPosition: {
            emit characterMovementBeganToDest(currentPosition, parsePosition(msg.destinationposition()), speed);
            break;
          }
          case proto::broadcast::CharacterMovementBegan::DestinationCase::kDestinationAngle: {
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
    case proto::broadcast::BroadcastMessage::BodyCase::kCharacterMovementEnded: {
        const proto::broadcast::CharacterMovementEnded &msg = message.charactermovementended();
        emit characterMovementEnded(parsePosition(msg.currentposition()));
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kCharacterNotMovingAngleChanged: {
        const proto::broadcast::CharacterNotMovingAngleChanged &msg = message.characternotmovinganglechanged();
        emit characterNotMovingAngleChanged(msg.angle());
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kRegionNameUpdate: {
        const proto::broadcast::RegionNameUpdate &msg = message.regionnameupdate();
        emit regionNameUpdate(msg.name());
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kItemUpdate: {
        const proto::broadcast::ItemUpdate &msg = message.itemupdate();
        std::optional<std::string> itemName;
        if (msg.has_itemname()) {
          itemName = msg.itemname();
        }
        if (msg.itemlocation() == proto::broadcast::ItemLocation::kCharacterInventory) {
          emit characterInventoryItemUpdate(msg.slotindex(), msg.quantity(), itemName);
        } else if (msg.itemlocation() == proto::broadcast::ItemLocation::kAvatarInventory) {
          emit avatarInventoryItemUpdate(msg.slotindex(), msg.quantity(), itemName);
        } else if (msg.itemlocation() == proto::broadcast::ItemLocation::kCosInventory) {
          emit cosInventoryItemUpdate(msg.slotindex(), msg.quantity(), itemName);
        } else if (msg.itemlocation() == proto::broadcast::ItemLocation::kStorage) {
          emit storageItemUpdate(msg.slotindex(), msg.quantity(), itemName);
        } else if (msg.itemlocation() == proto::broadcast::ItemLocation::kGuildStorage) {
          emit guildStorageItemUpdate(msg.slotindex(), msg.quantity(), itemName);
        }
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kEntitySpawned: {
        const proto::broadcast::EntitySpawned &msg = message.entityspawned();
        emit entitySpawned(msg.globalid(), parsePosition(msg.position()), msg.entity());
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kEntityDespawned: {
        const proto::broadcast::EntityDespawned &msg = message.entitydespawned();
        emit entityDespawned(msg.globalid());
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kEntityPositionChanged: {
        const proto::broadcast::EntityPositionChanged &msg = message.entitypositionchanged();
        emit entityPositionChanged(msg.globalid(), parsePosition(msg.position()));
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kEntityMovementBegan: {
        const proto::broadcast::EntityMovementBegan &msg = message.entitymovementbegan();
        const auto entityId = msg.globalid();
        const proto::broadcast::CharacterMovementBegan &charMovementMsg = msg.charactermovementbegan();
        sro::Position currentPosition = parsePosition(charMovementMsg.currentposition());
        const auto speed = charMovementMsg.speed();
        switch (charMovementMsg.destination_case()) {
          case proto::broadcast::CharacterMovementBegan::DestinationCase::kDestinationPosition: {
            emit entityMovementBeganToDest(entityId, currentPosition, parsePosition(charMovementMsg.destinationposition()), speed);
            break;
          }
          case proto::broadcast::CharacterMovementBegan::DestinationCase::kDestinationAngle: {
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
    case proto::broadcast::BroadcastMessage::BodyCase::kEntityMovementEnded: {
        const proto::broadcast::EntityMovementEnded &msg = message.entitymovementended();
        const proto::broadcast::CharacterMovementEnded &charMovementMsg = msg.charactermovementended();
        const auto entityId = msg.globalid();
        emit entityMovementEnded(entityId, parsePosition(charMovementMsg.currentposition()));
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kEntityLifeStateChanged: {
        const proto::broadcast::EntityLifeStateChanged &msg = message.entitylifestatechanged();
        const auto entityId = msg.globalid();
        const auto lifeState = lifeStateFromProto(msg.lifestate());
        emit entityLifeStateChanged(entityId, lifeState);
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kTrainingAreaSet: {
        const proto::broadcast::TrainingAreaSet &msg = message.trainingareaset();
        switch (msg.geometry_case()) {
          case proto::broadcast::TrainingAreaSet::GeometryCase::kCircle: {
            emit trainingAreaCircleSet(parsePosition(msg.circle().center()), msg.circle().radius());
            break;
          }
          default:
            throw std::runtime_error("Unhandled training area geometry");
        }
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kTrainingAreaReset: {
        emit trainingAreaReset();
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kStateMachineCreated: {
        const proto::broadcast::StateMachineCreated &msg = message.statemachinecreated();
        const auto name = msg.name();
        emit stateMachineCreated(name);
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kStateMachineDestroyed: {
        emit stateMachineDestroyed();
        break;
      }
    case proto::broadcast::BroadcastMessage::BodyCase::kWalkingPathUpdated: {
        const proto::broadcast::WalkingPathUpdated &msg = message.walkingpathupdated();
        std::vector<sro::Position> waypoints;
        waypoints.reserve(msg.waypoints_size());
        for (int i=0; i<msg.waypoints_size(); ++i) {
          waypoints.emplace_back(parsePosition(msg.waypoints(i)));
        }
        emit walkingPathUpdated(waypoints);
        break;
      }
    // case proto::broadcast::BroadcastMessage::BodyCase::kConfig: {
    //     const proto::broadcast::Config &msg = message.config();
    //     emit configReceived(msg.config());
    //   }
    default:
      // Unknown case. Might be a malformed message
      break;
  }
}