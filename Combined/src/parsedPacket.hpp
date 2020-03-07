#include "itemData.hpp"
#include "opcode.hpp"
#include "packetEnums.hpp"
#include "packetInnerStructures.hpp"
#include "shared/silkroad_security.h"

#include <string>

#ifndef PACKET_PARSING_HPP
#define PACKET_PARSING_HPP

namespace packet::parsing {

//=========================================================================================================================================================

class ParsedPacket {
protected:
  const Opcode opcode_;
public:
  ParsedPacket(const PacketContainer &packet);
  Opcode opcode() const;
  virtual ~ParsedPacket() = 0;
};

//=========================================================================================================================================================

class ParsedUnknown : public ParsedPacket {
public:
  ParsedUnknown(const PacketContainer &packet);
private:
};

//=========================================================================================================================================================

class ParsedServerHpMpUpdate : public ParsedPacket {
public:
  ParsedServerHpMpUpdate(const PacketContainer &packet);
  // packet_enums::WhichVital whichVital() const;
  uint8_t newValue() const;
private:
  uint32_t newValue_;
  // packet_enums::WhichVital whichVital_;
};

//=========================================================================================================================================================
class ParsedServerAgentCharacterData : public ParsedPacket {
public:
  ParsedServerAgentCharacterData(const PacketContainer &packet, const pk2::media::ItemData &itemData);
private:
};
  
//=========================================================================================================================================================

class ParsedServerAgentCharacterSelectionJoinResponse : public ParsedPacket {
public:
  ParsedServerAgentCharacterSelectionJoinResponse(const PacketContainer &packet);
  uint8_t result() const;
  uint16_t errorCode() const;
private:
  uint8_t result_;
  uint16_t errorCode_;
};

//=========================================================================================================================================================

class ParsedServerAgentCharacterSelectionActionResponse : public ParsedPacket {
public:
  ParsedServerAgentCharacterSelectionActionResponse(const PacketContainer &packet);
  packet_enums::CharacterSelectionAction action() const;
  uint8_t result() const;
  const std::vector<PacketInnerStructures::CharacterSelection::Character>& characters() const;
  uint16_t errorCode() const;
private:
  packet_enums::CharacterSelectionAction action_;
  uint8_t result_;
  std::vector<PacketInnerStructures::CharacterSelection::Character> characters_;
  uint16_t errorCode_; // TODO: Create enum for this
};

//=========================================================================================================================================================

class ParsedServerAuthResponse : public ParsedPacket {
public:
  ParsedServerAuthResponse(const PacketContainer &packet);
  uint8_t result() const;
  uint8_t errorCode() const;
private:
  uint8_t result_;
  uint8_t errorCode_;
};

//=========================================================================================================================================================

class ParsedLoginClientInfo : public ParsedPacket {
public:
  ParsedLoginClientInfo(const PacketContainer &packet);
  std::string serviceName() const;
private:
  std::string serviceName_;
};

//=========================================================================================================================================================

class ParsedLoginResponse : public ParsedPacket {
public:
  ParsedLoginResponse(const PacketContainer &packet);
  packet_enums::LoginResult result() const;
  uint32_t token() const;
private:
  packet_enums::LoginResult result_;
  uint32_t token_;
};

//=========================================================================================================================================================

class ParsedLoginServerList : public ParsedPacket {
public:
  ParsedLoginServerList(const PacketContainer &packet);
  uint16_t shardId() const;
private:
  uint16_t shardId_;
};

//=========================================================================================================================================================

class ParsedClientChat : public ParsedPacket {
public:
  ParsedClientChat(const PacketContainer &packet);
  packet_enums::ChatType chatType() const;
  uint8_t chatIndex() const;
  const std::string& receiverName() const;
  const std::string& message() const;
private:
  packet_enums::ChatType chatType_;
  uint8_t chatIndex_;
  std::string receiverName_;
  std::string message_;
};

//=========================================================================================================================================================

} // namespace packet::parsing

#endif // PACKET_PARSING_HPP