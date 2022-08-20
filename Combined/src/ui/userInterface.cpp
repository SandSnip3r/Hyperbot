#include "userInterface.hpp"

#include "ui-proto/request.pb.h"

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
  // TODO: std::string vs std::string_view. Might need to compile protobuf with c++17?
  characterNameUpdate.set_name(std::string(characterName));
  broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_characternameupdate() = characterNameUpdate;
  broadcast(broadcastMessage);
}

void UserInterface::broadcastGoldAmountUpdate(uint64_t goldAmount, broadcast::GoldLocation goldLocation) {
  broadcast::GoldAmountUpdate goldAmountUpdate;
  goldAmountUpdate.set_goldamount(goldAmount);
  goldAmountUpdate.set_goldlocation(goldLocation);
  broadcast::BroadcastMessage broadcastMessage;
  *broadcastMessage.mutable_goldamountupdate() = goldAmountUpdate;
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