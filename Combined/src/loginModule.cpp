#include "loginModule.hpp"
#include "opcode.hpp"
#include <iostream>

LoginModule::LoginModule(std::function<void(PacketContainer&, PacketContainer::Direction)> injectionFunction) : injectionFunction_(injectionFunction) {

}

void LoginModule::serverListReceived(const PacketContainer &packet) {
  StreamUtility stream = packet.data;
  uint8_t globalOpFlag = stream.Read<uint8_t>();
  while (globalOpFlag == 0x01) {
    // Read a "global op" , will be something like "SRO_Vietnam_TestLocal"
    uint8_t globalOpType = stream.Read<uint8_t>(); // For Atomix, its SRO_Taiwan_TestIn
    uint16_t globalNameLength = stream.Read<uint16_t>();
    std::cout << "globalNameLength: " << globalNameLength << '\n'; // Just checking endianness (TODO: Remove)
    std::string globalName;
    for (int i=0; i<globalNameLength; ++i) {
      globalName += stream.Read<uint8_t>();
    }
    std::cout << "LoginModule::serverListReceived: Read a global op name \"" << globalName << "\"\n";
    globalOpFlag = stream.Read<uint8_t>();
  }
  uint8_t shardFlag = stream.Read<uint8_t>();
  while (shardFlag == 0x01) {
    // Read a "shard" , will be something like "Atomix"
    shardId_ = stream.Read<uint16_t>();
    uint16_t shardNameLength = stream.Read<uint16_t>();
    std::cout << "shardNameLength: " << shardNameLength << '\n'; // Just checking endianness (TODO: Remove)
    std::string shardName;
    for (int i=0; i<shardNameLength; ++i) {
      shardName += stream.Read<uint8_t>();
    }
    std::cout << "LoginModule::serverListReceived: Read a shard name \"" << shardName << "\"\n";
    uint16_t shardCurrent = stream.Read<uint16_t>();
    uint16_t shardCapacity = stream.Read<uint16_t>();
    bool shardOnline = stream.Read<uint8_t>();
    std::cout << "LoginModule::serverListReceived: Server capacity is " << shardCurrent << '/' << shardCapacity << " and " << (shardOnline ? "is" : "isnt") << "online\n";
    uint8_t globalOpId = stream.Read<uint8_t>(); // Idk what this is, i guess globalOpType from above
    shardFlag = stream.Read<uint8_t>();
  }
  std::cout << "Our shard id is " << shardId_ << ", lets login now\n";

  //Build a LOGIN_CLIENT_AUTH packet
  StreamUtility loginAuthPacketData;
  loginAuthPacketData.Write<uint8_t>(kLocale_);
  loginAuthPacketData.Write<uint16_t>(kUsername_.size());
  loginAuthPacketData.Write_Ascii(kUsername_);
  loginAuthPacketData.Write<uint16_t>(kPassword_.size());
  loginAuthPacketData.Write_Ascii(kPassword_);
  loginAuthPacketData.Write<uint16_t>(shardId_);
  PacketContainer loginAuthPacket(static_cast<uint16_t>(Opcode::LOGIN_CLIENT_AUTH), loginAuthPacketData, 1, 0);
  //Inject the packet
  injectionFunction_(loginAuthPacket, PacketContainer::Direction::ClientToServer);
  loginPacketSent_ = true;
}

void LoginModule::serverAuthInfoReceived(const PacketContainer &packet) {
  if (!loginPacketSent_) {
    // We didnt cause this, the client must have. Nothing to do
    return;
  }
  StreamUtility stream = packet.data;
  uint8_t result = stream.Read<uint8_t>();
  if (result == 0x01) {
    // Success!
    loginToken_ = stream.Read<uint32_t>();
    gatewayLoginSuccessful_ = true;
    // Dont care about the rest, the proxy is handling that
  }
}

bool LoginModule::loginClientInfo(const PacketContainer &packet) {
  if (!loginPacketSent_ && !gatewayLoginSuccessful_) {
    // TODO: That if condition logic is bad. Handle failures better
    // We're not doing the login, let this be forwarded this to the client
    return true;
  }
  // Check to make sure this is the AgentServer and not GatewayServer
  StreamUtility stream = packet.data;
  uint16_t serverTypeStrLength = stream.Read<uint16_t>();
  std::string server = stream.Read_Ascii(serverTypeStrLength);
  if (server == "GatewayServer") {
    // Not what we want
    return true;
  }
  // Build a agentserver login packet
  //  4   uint    Token
  //  2   ushort  Username.Length
  //  *   string  Username
  //  2   ushort  Password.Length
  //  *   string  Password
  //  1   byte    Locale
  //  6   byte[]  MAC-Address
  StreamUtility clientAuthPacketData;
  clientAuthPacketData.Write<uint32_t>(loginToken_);
  clientAuthPacketData.Write<uint16_t>(kUsername_.size());
  clientAuthPacketData.Write_Ascii(kUsername_);
  clientAuthPacketData.Write<uint16_t>(kPassword_.size());
  clientAuthPacketData.Write_Ascii(kPassword_);
  clientAuthPacketData.Write<uint8_t>(kLocale_);
  //TODO: Handle mac address better
  clientAuthPacketData.Write<uint16_t>(0);
  clientAuthPacketData.Write<uint32_t>(macAddress_);
  PacketContainer clientAuthPacket(static_cast<uint16_t>(Opcode::CLIENT_AUTH), clientAuthPacketData, 1, 0);
  injectionFunction_(clientAuthPacket, PacketContainer::Direction::ClientToServer);
  return false;
}
