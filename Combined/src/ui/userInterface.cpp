#include "userInterface.hpp"

#include "ui-proto/request.pb.h"

namespace {

broadcast::EntityType sroEntityTypeToBroadcastEntityType(const sro::entity_types::EntityType entityType) {
  switch (entityType) {
    case sro::entity_types::EntityType::kSelf:
      return broadcast::EntityType::kSelf;
    case sro::entity_types::EntityType::kCharacter:
      return broadcast::EntityType::kCharacter;
    case sro::entity_types::EntityType::kPlayerCharacter:
      return broadcast::EntityType::kPlayerCharacter;
    case sro::entity_types::EntityType::kNonplayerCharacter:
      return broadcast::EntityType::kNonplayerCharacter;
    case sro::entity_types::EntityType::kMonster:
      return broadcast::EntityType::kMonster;
    case sro::entity_types::EntityType::kItem:
      return broadcast::EntityType::kItem;
    case sro::entity_types::EntityType::kPortal:
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
  broadcast::Position *currPos = broadcastMessage.mutable_charactermovementbegan()->mutable_currentposition();
  currPos->set_regionid(srcPosition.regionId());
  currPos->set_x(srcPosition.xOffset());
  currPos->set_y(srcPosition.yOffset());
  currPos->set_z(srcPosition.zOffset());
  broadcast::Position *destPos = broadcastMessage.mutable_charactermovementbegan()->mutable_destinationposition();
  destPos->set_regionid(destPosition.regionId());
  destPos->set_x(destPosition.xOffset());
  destPos->set_y(destPosition.yOffset());
  destPos->set_z(destPosition.zOffset());
  broadcastMessage.mutable_charactermovementbegan()->set_speed(speed);
  broadcast(broadcastMessage);
}

void UserInterface::broadcastMovementBeganUpdate(const sro::Position &srcPosition, uint16_t angle, float speed) {
  broadcast::BroadcastMessage broadcastMessage;
  broadcast::Position *currPos = broadcastMessage.mutable_charactermovementbegan()->mutable_currentposition();
  currPos->set_regionid(srcPosition.regionId());
  currPos->set_x(srcPosition.xOffset());
  currPos->set_y(srcPosition.yOffset());
  currPos->set_z(srcPosition.zOffset());
  broadcastMessage.mutable_charactermovementbegan()->set_destinationangle(angle);
  broadcastMessage.mutable_charactermovementbegan()->set_speed(speed);
  broadcast(broadcastMessage);
}

void UserInterface::broadcastMovementEndedUpdate(const sro::Position &currentPosition) {
  broadcast::BroadcastMessage broadcastMessage;
  broadcast::Position *currPos = broadcastMessage.mutable_charactermovementended()->mutable_currentposition();
  currPos->set_regionid(currentPosition.regionId());
  currPos->set_x(currentPosition.xOffset());
  currPos->set_y(currentPosition.yOffset());
  currPos->set_z(currentPosition.zOffset());
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

void UserInterface::broadcastEntitySpawned(uint32_t globalId, const sro::Position &position, sro::entity_types::EntityType entityType) {
  broadcast::BroadcastMessage broadcastMessage;
  auto *entitySpawnedMsg = broadcastMessage.mutable_entityspawned();

  entitySpawnedMsg->set_globalid(globalId);

  broadcast::Position *posMsg = entitySpawnedMsg->mutable_position();
  posMsg->set_regionid(position.regionId());
  posMsg->set_x(position.xOffset());
  posMsg->set_y(position.yOffset());
  posMsg->set_z(position.zOffset());

  entitySpawnedMsg->set_entitytype(sroEntityTypeToBroadcastEntityType(entityType));
  broadcast(broadcastMessage);
}

void UserInterface::broadcastEntityDespawned(uint32_t globalId) {
  broadcast::BroadcastMessage broadcastMessage;
  auto *entityDespawnedMsg = broadcastMessage.mutable_entitydespawned();
  entityDespawnedMsg->set_globalid(globalId);
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

} // namespace ui