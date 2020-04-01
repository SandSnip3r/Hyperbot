#include "chatEventModule.hpp"
#include "../packet/opcode.hpp"

#include <iostream>
#include <regex>
#include <Windows.h>

namespace module {

ChatEventModule::ChatEventModule(std::function<void(PacketContainer&, PacketContainer::Direction)> injectionFunction) : injectionFunction_(injectionFunction) {

}

void ChatEventModule::writeAllChatMessage(const std::string &message) {
  // 1   byte    chatType
  // 1   byte    chatIndex
  // if(chatType == ChatType.PM)
  // {
  //     2   ushort  reciver.Length
  //     *   string  reciver
  // }
  // 2   ushort  message.Length
  // *   string  message
  StreamUtility allChatPacketData;
  allChatPacketData.Write<uint8_t>(static_cast<uint8_t>(ChatType::kAll));
  allChatPacketData.Write<uint8_t>(chatIndex);
  allChatPacketData.Write<uint16_t>(message.size());
  allChatPacketData.Write_Ascii(message);
  PacketContainer allChatPacket(static_cast<uint16_t>(packet::Opcode::CLIENT_CHAT), allChatPacketData, 1, 0);
  injectionFunction_(allChatPacket, PacketContainer::Direction::kClientToServer);
}

void ChatEventModule::serverChatReceived(const PacketContainer &packet) {
  std::cout << "ChatEventModule::serverChatReceived\n";
  StreamUtility stream = packet.data;
  // [1559663320906]  (S->C)0,0,0,3026 02 0a 00 53 61 6e 64 53 6e 69 70 33 72 07 00 68 65 79 20 66 61 6d
                                              // S  a  n  d  S  n  i  p  3  r        h  e  y     f  a  m
  // 1   byte    chatType
  // if(chatType == ChatType.All ||
  //   chatType == ChatType.AllGM ||
  //   chatType == ChatType.NPC)
  // {
  //     4   uint    message.Sender.UniqueID
  // }
  // else if(chatType == ChatType.PM ||
  //         chatType == ChatType.Party ||
  //         chatType == ChatType.Guild ||
  //         chatType == ChatType.Global ||
  //         chatType == ChatType.Stall ||
  //         chatType == ChatType.Union ||
  //         chatType == ChatType.Accademy)        
  // {
  //     2   ushort  message.Sender.Name.Length
  //     *   string  message.Sender.Name
  // }
  // 2   ushort  message.Length
  // *   string  message
  uint8_t chatType = stream.Read<uint8_t>();
  if (chatType == static_cast<uint8_t>(ChatType::kNotice)) {
    std::cout << "Valid chat type\n";
    // This is a chat type we care about
    const uint16_t kMessageLength = stream.Read<uint16_t>();
    const std::string kMessage = stream.Read_Ascii(kMessageLength);
    std::cout << "Received message \"" << kMessage << "\"\n";
    std::regex lotteryRegex(R"delim(\[Lottery\] Event has started, type 'register' in your all chat to sign in the lottery\.)delim");
    std::regex firstTypeRegex(R"delim(\[First Type\] Event has started the word is: \[(.*)\])delim");
    std::smatch matchResults;
    bool attempted=false;
    if (std::regex_search(kMessage, matchResults, lotteryRegex)) {
      if (matchResults.ready()) {
        attempted = true;
        std::cout << "A lottery! Lets enter the lottery\n";
        // Lottery event happening!
        // Send an all chat message to enter
        writeAllChatMessage("register");
      }
    } else if (std::regex_search(kMessage, matchResults, firstTypeRegex)) {
      if (matchResults.ready() && matchResults.size() == 2) {
        attempted = true;
        const std::string kTextToWrite = matchResults[1];
        std::cout << "A \"First Type\" event! Lets win. Gotta write \"" << kTextToWrite << "\" to all chat\n";
        // First Type event happening!
        // Send an all chat message to win
        writeAllChatMessage(kTextToWrite);
      }
    } else {
      std::cout << "Message did not match any regex\n";
    }
    if (attempted) {
      ++attempts;
      std::cout << "Attempt #" << attempts << '\n';
    }
  }
}

// [09:45:15] Global notice:[Lottery] Event will start in 1 minute, get ready...
// [09:46:15] Global notice:[Lottery] Event has started, type 'register' in your all chat to sign in the lottery.
// [09:46:45] Global notice:[Cursed] has won the lottery event.
// [09:46:50] Global notice:[Lottery] Event has finished, better luck next time.

// [17:55:12] Global notice:[Lottery] Event will start in 1 minute, get ready...
// [17:56:12] Global notice:[Lottery] Event has started, type 'register' in your all chat to sign in the lottery.
// [17:56:17] Global notice:You have been successfully registered to the lottery event, please wait for the results.
// [17:56:42] Global notice:[AhmedRabie] has won the lottery event.
// [05:08:29] Global notice:[Lottery] Event will start in 1 minute, get rea
// dy...
// [05:09:29] Global notice:[Lottery] Event has started, type 'register' in your all chat to sign in the lottery.
// [05:09:59] Global notice:[Surrender] has won the lottery event.
// [05:10:04] Global notice:[Lottery] Event has finished, better luck next time.

//=============================================================================================================

// [08:44:13] Global notice:[First Type] Event will start in 1 minute, please write the answer in  your (All chat), get ready...
// [08:45:13] Global notice:[First Type] Event has started the word is: [PYKNic]
// [08:45:15] Global notice:[Elektra] has won the first-type event with his answer [PYKNic]
// [08:46:13] Global notice:[First Type] Event has finished, better luck next time.

// [15:53:07] Global notice:[First Type] Event will start in 1 minute, please write the answer in  your (All chat), get ready...
// [15:54:07] Global notice:[First Type] Event has started the word is: [DiGiTiGrADiEnt]
// [15:54:09] Global notice:[Darja] has won the first-type event with his answer [DiGiTiGrADiEnt]
// [15:55:07] Global notice:[First Type] Event has finished, better luck next time.

// [16:54:09] Global notice:[First Type] Event will start in 1 minute, please write the answer in  your (All chat), get ready...
// [16:55:09] Global notice:[First Type] Event has started the word is: [pasSIUnCLE]
// [16:55:12] Global notice:[Krikko] has won the first-type event with his answer [pasSIUnCLE]
// [16:56:09] Global notice:[First Type] Event has finished, better luck next time.

//=============================================================================================================

// [06:25:42] Global notice:[Lucky Party Number] Event will start in 1 minute, get ready...
// [06:26:42] Global notice:[Lucky Party Number] Event has started, first player to create party number [891] wins.
// [06:26:49] Global notice:[Klong] has won the Lucky Party Number event.
// [06:27:15] Global notice:[Lucky Party Number] Event has finished, better luck next time.

// [11:47:58] Global notice:[Lucky Party Number] Event will start in 1 minute, get ready...
// [11:48:58] Global notice:[Lucky Party Number] Event has started, first player to create party number [145] wins.
// [11:49:07] Global notice:[MaryJane] has won the Lucky Party Number event.
// [11:49:31] Global notice:[Lucky Party Number] Event has finished, better luck next time.

// [12:49:07] Global notice:[Lucky Party Number] Event will start in 1 minute, get ready...
// [12:50:07] Global notice:[Lucky Party Number] Event has started, first player to create party number [188] wins.
// [12:50:10] Global notice:[Gremlinseye] has won the Lucky Party Number event.
// [12:50:40] Global notice:[Lucky Party Number] Event has finished, better luck next time.

//=============================================================================================================

/* [07:26:49] Global notice:[Alchemy] Event will start in 1 minute, get ready...
[07:27:49] Global notice:[Alchemy] Event has started, try to plus an item to plus: [2]
[07:27:54] Global notice:[Moka_] has pimped an item to [plus 2] and won the alchemy event.
[07:30:49] Global notice:[Alchemy] Event has finished, better luck next time.
[10:46:50] Global notice:[Trivia] Event will start in 1 minute, please write the answer in  your (All chat), get ready...
[10:47:50] Global notice:[Trivia] Event has started the question is: [Are cougars are herbivores or carnivores ?]
[10:47:56] Global notice:[Duo] has won the trivia event with his answer [carnivores]
[10:48:20] Global notice:[Trivia] Event has finished, better luck next time.
[13:50:11] Global notice:[Math] Event will start in 1 minute, please write the answer in  your (All chat), get ready...
[13:51:11] Global notice:[Math] Round started the question is: [(468 minus 877)]
[13:51:17] Global notice:[Elektra] has won the math event with his answer [-409]
[13:51:41] Global notice:[Math] Event has finished, better luck next time.
[14:51:17] Global notice:[Re-arrange] Event will start in 1 minute, please write the answer in  your (All chat), get ready...
[14:52:17] Global notice:[Re-arrange] Event has started the word is: tasemettn
[14:53:07] Global notice:[Awe4k] has won the re-arrange event with his answer [statement]
[14:54:17] Global notice:Re-arrange] Event has finished, better luck next time.
[17:56:47] Global notice:[Lottery] Event has finished, better luck next time.
[18:56:47] Global notice:[Alchemy] Event will start in 1 minute, get ready...
[18:57:47] Global notice:[Alchemy] Event has started, try to plus an item to plus: [3]
[18:58:05] Global notice:[Senox] has pimped an item to [plus 3] and won the alchemy event.
[19:00:47] Global notice:[Alchemy] Event has finished, better luck next time.
[19:58:05] Global notice:[Trivia] Event will start in 1 minute, please write the answer in  your (All chat), get ready...
[19:59:05] Global notice:[Trivia] Event has started the question is: [Emerald is the birthstone for which month ?]
[19:59:27] Global notice:[EnjoySilence] has won the trivia event with his answer [may]
[19:59:35] Global notice:[Trivia] Event has finished, better luck next time.
[20:59:27] Global notice:[Alchemy] Event will start in 1 minute, get ready...
[21:00:27] Global notice:[Alchemy] Event has started, try to plus an item to plus: [3]
[21:00:41] Global notice:[Moka_] has pimped an item to [plus 3] and won the alchemy event.
[21:03:27] Global notice:[Alchemy] Event has finished, better luck next time.
[22:00:41] Global notice:[Lucky Party Number] Event will start in 1 minute, get ready...
[22:01:41] Global notice:[Lucky Party Number] Event has started, first player to create party number [426] wins.
[22:01:48] Global notice:[Dead_Aim] has won the Lucky Party Number event.
[22:02:14] Global notice:[Lucky Party Number] Event has finished, better luck next time.
[23:01:49] Global notice:[Alchemy] Event will start in 1 minute, get ready...
[23:02:49] Global notice:[Alchemy] Event has started, try to plus an item to plus: [2]
[23:03:06] Global notice:[AhmedRabie] has pimped an item to [plus 2] and won the alchemy event.
[23:05:49] Global notice:[Alchemy] Event has finished, better luck next time.
[00:03:06] Global notice:[Math] Event will start in 1 minute, please write the answer in  your (All chat), get ready...
[00:04:06] Global notice:[Math] Round started the question is: [(394 minus 85)]
[00:04:13] Global notice:[Joo] has won the math event with his answer [309]
[00:04:36] Global notice:[Math] Event has finished, better luck next time.
[01:04:13] Global notice:[Lucky Party Number] Event will start in 1 minute, get ready...
[01:05:13] Global notice:[Lucky Party Number] Event has started, first player to create party number [478] wins.
[01:05:19] Global notice:[Sneeky] has won the Lucky Party Number event.
[01:05:46] Global notice:[Lucky Party Number] Event has finished, better luck next time.
[02:05:19] Global notice:[Re-arrange] Event will start in 1 minute, please write the answer in  your (All chat), get ready...
[02:06:19] Global notice:[Re-arrange] Event has started the word is: eatb
[02:06:23] Global notice:[TrongXeTank] has won the re-arrange event with his answer [beat]
[02:08:19] Global notice:Re-arrange] Event has finished, better luck next time.
[03:06:23] Global notice:[Re-arrange] Event will start in 1 minute, please write the answer in  your (All chat), get ready...
[03:07:23] Global notice:[Re-arrange] Event has started the word is: drac
[03:07:26] Global notice:[SoSicK] has won the re-arrange event with his answer [card]
[03:09:23] Global notice:Re-arrange] Event has finished, better luck next time.
[04:07:26] Global notice:[Lucky Party Number] Event will start in 1 minute, get ready...
[04:08:26] Global notice:[Lucky Party Number] Event has started, first player to create party number [554] wins.
[04:08:29] Global notice:[IM_Top] has won the Lucky Party Number event.
[04:08:59] Global notice:[Lucky Party Number] Event has finished, better luck next time.
[06:07:54] Global notice:[Brodah] has successfully enhanced [Probound Iranggjingun Pike] to [plus 7]
[06:10:04] Global notice:[Lucky Party Number] Event will start in 1 minute, get ready...
[06:11:04] Global notice:[Lucky Party Number] Event has started, first player to create party number [611] wins.
[06:11:13] Global notice:[EnjoySilence] has won the Lucky Party Number event.
[06:11:37] Global notice:[Lucky Party Number] Event has finished, better luck next time.
[07:11:13] Global notice:[Alchemy] Event will start in 1 minute, get ready...
[07:12:13] Global notice:[Alchemy] Event has started, try to plus an item to plus: [2]
[07:12:19] Global notice:[EnjoySilence] has pimped an item to [plus 2] and won the alchemy event.
[07:15:13] Global notice:[Alchemy] Event has finished, better luck next time.
[07:18:47] Global notice:[Arc] has successfully enhanced [Genuine Master Sword] to [plus 7]
[08:12:19] Global notice:[Math] Event will start in 1 minute, please write the answer in  your (All chat), get ready...
[08:13:19] Global notice:[Math] Round started the question is: [(927 multiplied by 107)]
[08:13:26] Global notice:[nForce] has won the math event with his answer [99189]
[08:13:50] Global notice:[Math] Event has finished, better luck next time. */

} // namespace module