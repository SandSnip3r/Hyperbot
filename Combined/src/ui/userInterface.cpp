#include "userInterface.hpp"

#include "ui-proto/request.pb.h"

namespace {

broadcast::EntityType sroEntityTypeToBroadcastEntityType(const entity::EntityType entityType) {
  switch (entityType) {
    case entity::EntityType::kSelf:
      return broadcast::EntityType::kSelf;
    case entity::EntityType::kCharacter:
      return broadcast::EntityType::kCharacter;
    case entity::EntityType::kPlayerCharacter:
      return broadcast::EntityType::kPlayerCharacter;
    case entity::EntityType::kNonplayerCharacter:
      return broadcast::EntityType::kNonplayerCharacter;
    case entity::EntityType::kMonster:
      return broadcast::EntityType::kMonster;
    case entity::EntityType::kItem:
      return broadcast::EntityType::kItem;
    case entity::EntityType::kPortal:
      return broadcast::EntityType::kPortal;
    default:
      throw std::runtime_error("Unknown entity type");
  }
}

} // anonymous namespace

namespace ui {

UserInterface::UserInterface(broker::EventBroker &eventBroker) : eventBroker_(eventBroker) {}

UserInterface::~UserInterface() {
  thr_.join();
}

void UserInterface::run() {
  // Set up publisher
  publisher_.bind("tcp://*:5556");

  // Run the request receiver in another thread
  thr_ = std::thread(&UserInterface::privateRun, this);
}

void UserInterface::broadcastCharacterSpawn() {
  broadcast::BroadcastMessage broadcastMessage;
  broadcastMessage.mutable_characterspawn();
  broadcast(broadcastMessage);
}

void UserInterface::broadcastCharacterHpUpdate(uint32_t currentHp) {
  broadcast::CharacterHpUpdate characterHpUpdate;
  characterHpUpdate.set_currenthp(currentHp);
  broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_characterhpupdate() = characterHpUpdate;
  broadcast(broadcastMessage);
}

void UserInterface::broadcastCharacterMpUpdate(uint32_t currentMp) {
  broadcast::CharacterMpUpdate characterMpUpdate;
  characterMpUpdate.set_currentmp(currentMp);
  broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_charactermpupdate() = characterMpUpdate;
  broadcast(broadcastMessage);
}

void UserInterface::broadcastCharacterMaxHpMpUpdate(uint32_t maxHp, uint32_t maxMp) {
  broadcast::CharacterMaxHpMpUpdate characterMaxHpMpUpdate;
  characterMaxHpMpUpdate.set_maxhp(maxHp);
  characterMaxHpMpUpdate.set_maxmp(maxMp);
  broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_charactermaxhpmpupdate() = characterMaxHpMpUpdate;
  broadcast(broadcastMessage);
}

void UserInterface::broadcastCharacterLevelUpdate(uint8_t currentLevel, int64_t expRequired) {
  broadcast::CharacterLevelUpdate characterLevelUpdate;
  characterLevelUpdate.set_level(currentLevel);
  characterLevelUpdate.set_exprequired(expRequired);
  broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_characterlevelupdate() = characterLevelUpdate;
  broadcast(broadcastMessage);
}

void UserInterface::broadcastCharacterExperienceUpdate(uint64_t currentExperience, uint32_t currentSpExperience) {
  broadcast::CharacterExperienceUpdate characterExperienceUpdate;
  characterExperienceUpdate.set_currentexperience(currentExperience);
  characterExperienceUpdate.set_currentspexperience(currentSpExperience);
  broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_characterexperienceupdate() = characterExperienceUpdate;
  broadcast(broadcastMessage);
}

void UserInterface::broadcastCharacterSpUpdate(uint32_t skillPoints) {
  broadcast::CharacterSpUpdate characterSpUpdate;
  characterSpUpdate.set_skillpoints(skillPoints);
  broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_characterspupdate() = characterSpUpdate;
  broadcast(broadcastMessage);
}

void UserInterface::broadcastCharacterNameUpdate(std::string_view characterName) {
  broadcast::CharacterNameUpdate characterNameUpdate;
  characterNameUpdate.set_name(std::string(characterName));
  broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_characternameupdate() = characterNameUpdate;
  broadcast(broadcastMessage);
}

void UserInterface::broadcastGoldAmountUpdate(uint64_t goldAmount, broadcast::ItemLocation goldLocation) {
  broadcast::GoldAmountUpdate goldAmountUpdate;
  goldAmountUpdate.set_goldamount(goldAmount);
  goldAmountUpdate.set_goldlocation(goldLocation);
  broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_goldamountupdate() = goldAmountUpdate;
  broadcast(broadcastMessage);
}

void UserInterface::broadcastMovementBeganUpdate(const sro::Position &srcPosition, const sro::Position &destPosition, float speed) {
  broadcast::BroadcastMessage broadcastMessage;
  setCharacterMovementBegan(broadcastMessage.mutable_charactermovementbegan(), srcPosition, destPosition, speed);
  broadcast(broadcastMessage);
}

void UserInterface::broadcastMovementBeganUpdate(const sro::Position &srcPosition, uint16_t angle, float speed) {
  broadcast::BroadcastMessage broadcastMessage;
  setCharacterMovementBegan(broadcastMessage.mutable_charactermovementbegan(), srcPosition, angle, speed);
  broadcast(broadcastMessage);
}

void UserInterface::broadcastMovementEndedUpdate(const sro::Position &currentPosition) {
  broadcast::BroadcastMessage broadcastMessage;
  setCharacterMovementEnded(broadcastMessage.mutable_charactermovementended(), currentPosition);
  broadcast(broadcastMessage);
}

void UserInterface::broadcastRegionNameUpdate(std::string_view regionName) {
  broadcast::RegionNameUpdate regionNameUpdate;
  regionNameUpdate.set_name(std::string(regionName));
  broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_regionnameupdate() = regionNameUpdate;
  broadcast(broadcastMessage);
}

void UserInterface::broadcastItemUpdate(broadcast::ItemLocation itemLocation, uint8_t slotIndex, uint16_t quantity, std::optional<std::string> itemName) {
  broadcast::BroadcastMessage broadcastMessage;
  auto *itemUpdate = broadcastMessage.mutable_itemupdate();
  itemUpdate->set_itemlocation(itemLocation);
  itemUpdate->set_slotindex(slotIndex);
  itemUpdate->set_quantity(quantity);
  if (quantity != 0) {
    if (!itemName) {
      throw std::runtime_error("Quantity of item positive, but we were given no item name");
    }
    itemUpdate->set_itemname(*itemName);
  }
  broadcast(broadcastMessage);
}

void UserInterface::broadcastEntitySpawned(uint32_t globalId, const sro::Position &position, entity::EntityType entityType) {
  broadcast::BroadcastMessage broadcastMessage;
  auto *entitySpawnedMsg = broadcastMessage.mutable_entityspawned();
  entitySpawnedMsg->set_globalid(globalId);
  setPosition(entitySpawnedMsg->mutable_position(), position);
  entitySpawnedMsg->set_entitytype(sroEntityTypeToBroadcastEntityType(entityType));
  broadcast(broadcastMessage);
}

void UserInterface::broadcastEntityDespawned(uint32_t globalId) {
  broadcast::BroadcastMessage broadcastMessage;
  auto *entityDespawnedMsg = broadcastMessage.mutable_entitydespawned();
  entityDespawnedMsg->set_globalid(globalId);
  broadcast(broadcastMessage);
}

void UserInterface::broadcastEntityPositionChanged(const sro::scalar_types::EntityGlobalId globalId, const sro::Position &position) {
  broadcast::BroadcastMessage broadcastMessage;
  auto *entityPositionChangedMsg = broadcastMessage.mutable_entitypositionchanged();
  entityPositionChangedMsg->set_globalid(globalId);
  setPosition(entityPositionChangedMsg->mutable_position(), position);
  broadcast(broadcastMessage);
}

void UserInterface::broadcastEntityMovementBegan(const sro::scalar_types::EntityGlobalId globalId, const sro::Position &srcPosition, const sro::Position &destPosition, float speed) {
  broadcast::BroadcastMessage broadcastMessage;
  broadcast::EntityMovementBegan *entityMovementBegan = broadcastMessage.mutable_entitymovementbegan();
  entityMovementBegan->set_globalid(globalId);
  setCharacterMovementBegan(entityMovementBegan->mutable_charactermovementbegan(), srcPosition, destPosition, speed);
  broadcast(broadcastMessage);
}

void UserInterface::broadcastEntityMovementBegan(const sro::scalar_types::EntityGlobalId globalId, const sro::Position &srcPosition, uint16_t angle, float speed) {
  broadcast::BroadcastMessage broadcastMessage;
  broadcast::EntityMovementBegan *entityMovementBegan = broadcastMessage.mutable_entitymovementbegan();
  entityMovementBegan->set_globalid(globalId);
  setCharacterMovementBegan(entityMovementBegan->mutable_charactermovementbegan(), srcPosition, angle, speed);
  broadcast(broadcastMessage);
}

void UserInterface::broadcastEntityMovementEnded(const sro::scalar_types::EntityGlobalId globalId, const sro::Position &currentPosition) {
  broadcast::BroadcastMessage broadcastMessage;
  broadcast::EntityMovementEnded *entityMovementEnded = broadcastMessage.mutable_entitymovementended();
  entityMovementEnded->set_globalid(globalId);
  setCharacterMovementEnded(entityMovementEnded->mutable_charactermovementended(), currentPosition);
  broadcast(broadcastMessage);
}

void UserInterface::broadcast(const broadcast::BroadcastMessage &broadcastProto) {
  zmq::message_t message;
  message.rebuild(broadcastProto.ByteSizeLong());
  broadcastProto.SerializeToArray(message.data(), message.size());
  const auto res = publisher_.send(message, zmq::send_flags::none);
  // TODO: Check result

  // Old method:
  // zmq::message_t msg;
  // std::string str;
  // broadcastMessage.SerializeToString(&str);
  // msg.rebuild(str.data(), str.size());
  // auto res = publisher_.send(msg, zmq::send_flags::none);
}

void UserInterface::privateRun() {
  // Run request receiver
  zmq::socket_t socket(context_, zmq::socket_type::rep);
  socket.bind("tcp://*:5555");
  while (1) {
    // Wait for a request
    zmq::message_t request;
    socket.recv(request, zmq::recv_flags::none);

    handle(request);
    
    // Immediately respond with an acknowledgement
    const std::string response{"ack"};
    socket.send(zmq::buffer(response), zmq::send_flags::none);
  }
}

void UserInterface::handle(const zmq::message_t &request) {
  // Parse the request
  request::RequestMessage requestMsg;
  requestMsg.ParseFromArray(request.data(), request.size());
  switch (requestMsg.body_case()) {
    case request::RequestMessage::BodyCase::kPacketData: {
        const request::PacketToInject &packet = requestMsg.packetdata();
        const event::InjectPacket::Direction dir = (packet.direction() == request::PacketToInject::kClientToServer) ? event::InjectPacket::Direction::kClientToServer : event::InjectPacket::Direction::kServerToClient;
        eventBroker_.publishEvent(std::make_unique<event::InjectPacket>(dir, packet.opcode(), packet.data()));
        break;
      }
    case request::RequestMessage::BodyCase::kDoAction: {
        const request::DoAction &doActionMsg = requestMsg.doaction();
        if (doActionMsg.action() == request::DoAction::kStartTraining) {
          eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStartTraining));
        } else if (doActionMsg.action() == request::DoAction::kStopTraining) {
          eventBroker_.publishEvent(std::make_unique<event::Event>(event::EventCode::kStopTraining));
        }
        break;
      }
    default:
      std::cout << "Unknown request type" << std::endl;
      break;
  }
}

void UserInterface::setPosition(broadcast::Position *msg, const sro::Position &pos) const {
  msg->set_regionid(pos.regionId());
  msg->set_x(pos.xOffset());
  msg->set_y(pos.yOffset());
  msg->set_z(pos.zOffset());
}

void UserInterface::setCharacterMovementBegan(broadcast::CharacterMovementBegan *msg, const sro::Position &srcPosition, const sro::Position &destPosition, const float speed) const {
  setPosition(msg->mutable_currentposition(), srcPosition);
  setPosition(msg->mutable_destinationposition(), destPosition);
  msg->set_speed(speed);
}

void UserInterface::setCharacterMovementBegan(broadcast::CharacterMovementBegan *msg, const sro::Position &srcPosition, const sro::MovementAngle angle, const float speed) const {
  setPosition(msg->mutable_currentposition(), srcPosition);
  msg->set_destinationangle(angle);
  msg->set_speed(speed);
}

void UserInterface::setCharacterMovementEnded(broadcast::CharacterMovementEnded *msg, const sro::Position &currentPosition) const {
  setPosition(msg->mutable_currentposition(), currentPosition);
}

} // namespace ui