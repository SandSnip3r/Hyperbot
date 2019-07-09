#include "shared/silkroad_security.h"
#include "packetEnums.hpp"
#include "packetInnerStructures.hpp"

#include <string>

#ifndef PACKET_PARSING_HPP
#define PACKET_PARSING_HPP

namespace PacketParsing {

class PacketParser {
protected:
  bool parsed_{false};
  void parsedCheck();
  virtual void parsePacket() = 0;
public:
  virtual ~PacketParser();
};

PacketParser* newPacketParser(const PacketContainer &packet);

class UnknownPacket : public PacketParser {
private:
  const PacketContainer &packet_;
  void parsePacket() override;
public:
  UnknownPacket(const PacketContainer &packet);
  Opcode opcode() const;
private:
};

class ServerAgentCharacterDataPacket : public PacketParser {
private:
  const PacketContainer &packet_;
  void parsePacket() override;
public:
  ServerAgentCharacterDataPacket(const PacketContainer &packet);
private:
};
  
class ServerAgentCharacterSelectionJoinResponsePacket : public PacketParser {
private:
  const PacketContainer &packet_;
  void parsePacket() override;
public:
  ServerAgentCharacterSelectionJoinResponsePacket(const PacketContainer &packet);
  uint8_t result();
  uint16_t errorCode();
private:
  uint8_t result_;
  uint16_t errorCode_;
};

class ServerAgentCharacterSelectionActionResponsePacket : public PacketParser {
private:
  const PacketContainer &packet_;
  void parsePacket() override;
public:
  ServerAgentCharacterSelectionActionResponsePacket(const PacketContainer &packet);
  PacketEnums::CharacterSelectionAction action();
  uint8_t result();
  const std::vector<PacketInnerStructures::CharacterSelection::Character>& characters();
  uint16_t errorCode();
private:
  PacketEnums::CharacterSelectionAction action_;
  uint8_t result_;
  std::vector<PacketInnerStructures::CharacterSelection::Character> characters_;
  uint16_t errorCode_; // TODO: Create enum for this
};

class ServerAuthResponsePacket : public PacketParser {
private:
  const PacketContainer &packet_;
  void parsePacket() override;
public:
  ServerAuthResponsePacket(const PacketContainer &packet);
  uint8_t result();
  uint8_t errorCode();
private:
  uint8_t result_;
  uint8_t errorCode_;
};

class LoginClientInfoPacket : public PacketParser {
private:
  const PacketContainer &packet_;
  void parsePacket() override;
public:
  LoginClientInfoPacket(const PacketContainer &packet);
  std::string serviceName();
private:
  std::string serviceName_;
};

class LoginResponsePacket : public PacketParser {
private:
  const PacketContainer &packet_;
  void parsePacket() override;
public:
  LoginResponsePacket(const PacketContainer &packet);
  PacketEnums::LoginResult result();
  uint32_t token();
private:
  PacketEnums::LoginResult result_;
  uint32_t token_;
};

class LoginServerListPacket : public PacketParser {
private:
  const PacketContainer &packet_;
  void parsePacket() override;
public:
  LoginServerListPacket(const PacketContainer &packet);
  uint16_t shardId();
private:
  uint16_t shardId_;
};

class ClientCafePacket : public PacketParser {
private:
  const PacketContainer &packet_;
  void parsePacket() override;
public:
  ClientCafePacket(const PacketContainer &packet);
private:
};

class ClientChatPacket : public PacketParser {
private:
  const PacketContainer &packet_;
  void parsePacket() override;
public:
  ClientChatPacket(const PacketContainer &packet);
  PacketEnums::ChatType chatType();
  uint8_t chatIndex();
  const std::string& receiverName();
  const std::string& message();
private:
  PacketEnums::ChatType chatType_;
  uint8_t chatIndex_;
  std::string receiverName_;
  std::string message_;
};

// class ServerCharacterPacket : public PacketParser {
// private:
//   const PacketContainer &packet_;
//   int maxHp_;
//   void parsePacket() override;
// public:
//   ServerCharacterPacket(const PacketContainer &packet);
//   int maxHp();
// };

// class ServerHpMpUpdatePacket : public PacketParser {
// private:
//   const PacketContainer &packet_;
//   int hp_;
//   void parsePacket() override;
// public:
//   ServerHpMpUpdatePacket(const PacketContainer &packet);
//   int hp();
// };

} // namespace PacketParsing

#endif // PACKET_PARSING_HPP