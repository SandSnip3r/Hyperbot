#include "bot.hpp"

Bot::Bot(const config::CharacterLoginData &loginData,
         const pk2::GameData &gameData,
         broker::PacketBroker &broker) :
      loginData_(loginData),
      gameData_(gameData),
      broker_(broker) {
  eventBroker_.run();
  userInterface_.run();
}

// bool Bot::handleClientChat(std::unique_ptr<PacketParsing::PacketParser> &packetParser) {
//   std::cout << "Bot::handleClientChat\n";
//   PacketParsing::ClientChatPacket *clientChat = dynamic_cast<PacketParsing::ClientChatPacket*>(packetParser.get());
//   if (clientChat == nullptr) {
//     std::cerr << "Bot::handleClientChat Invalid PacketParser type\n";
//   }

//   const std::string kChatCommandPrefix{"\\b "};
//   if (clientChat->chatType() == packet::enums::ChatType::kAll) {
//     // All chat
//     const std::string &msg = clientChat->message();
//     std::cout << "all chat msg \"" << msg << "\"\n";
//     if (msg.find(kChatCommandPrefix) == 0) {
//       std::string command = msg.substr(kChatCommandPrefix.size());
//       commandHandler(command);
//       return false;
//     }
//   }
//   return true;
// }

// void Bot::commandHandler(const std::string &command) {
//   std::cout << "Bot::commandHandler Given command \"" << command << "\"\n";

//   const std::string kNoticeMessage = "Command \"" + command + "\" accepted.";

//   auto noticePacket = packet::building::ServerAgentChatUpdate::notice(kNoticeMessage);
//   broker_.injectPacket(noticePacket, PacketContainer::Direction::kServerToClient);
  
//   const int kRegionId = 0x60a9;
//   const int kXOffset = 0x00e8;
//   const int kYOffset = 0x0000;
//   const int kZOffset = 0x0772;

//   auto movePacket = packet::building::ClientAgentCharacterMoveRequest::packet(kRegionId, kXOffset, kYOffset, kZOffset);
//   broker_.injectPacket(movePacket, PacketContainer::Direction::kClientToServer);
// }