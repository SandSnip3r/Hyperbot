# Packet serialization and deserialization

My structure has two interface classes: `PacketParser` (to help with deserialization) and `PacketBuilder` (to help with serialization)

## Deserialization

- At the point where packets are received from the network and ready to be passed to the business logic, a fcuntion `newPacketParser` is called
- The signature of the function is `PacketParser* newPacketParser(PacketContainer packet)`
  - `PacketContainer` is Weeman's original structure which is just an opcode an an array of raw bytes
- `newPacketParser` returns the proper deriving `PacketParser` class depending on the opcode
- The derivatives of `PacketParser` contain member data sufficient to hold all possible branchings of the packet parsing

Here's an example of a class deriving from `PacketParser`:
```cpp
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
```

From the packet handler's perspective, usage looks like this:
```cpp
void Bot::handleClientChat(std::unique_ptr<PacketParsing::PacketParser> &packetParser) {
  PacketParsing::ClientChatPacket *clientChat = dynamic_cast<PacketParsing::ClientChatPacket*>(packetParser.get());
  if (clientChat->chatType() == PacketEnums::ChatType::kAll) {
    const std::string &msg = clientChat->message();
    std::cout << "Player said \"" << msg << "\" in all chat\n";
  }
}
```

## Serialization

- Again, there's an interface class `PacketBuilder` which is inherited from for each packet we need to build
- The derivatives of `PacketBuilder` contain multiple constructors, each to allow for all possible branchings when building packets

Here's an example of a class deriving from `PacketBuilder`:
```cpp
class ServerChatPacketBuilder : public PacketBuilder {
private:
  PacketEnums::ChatType chatType_;
  uint32_t senderId_;
  std::string senderName_;
  std::string message_;
public:
  ServerChatPacketBuilder(PacketEnums::ChatType chatType, const std::string &message);
  ServerChatPacketBuilder(PacketEnums::ChatType chatType, uint32_t senderId, const std::string &message);
  ServerChatPacketBuilder(PacketEnums::ChatType chatType, const std::string &senderName, const std::string &message);
  PacketContainer packet() const;
};
```

When you need to build a packet, you just create the object like this:
```cpp
const std::string kNoticeMessage = "I am a GM";
auto noticePacket = PacketBuilding::ServerChatPacketBuilder(PacketEnums::ChatType::kNotice, kNoticeMessage).packet();
broker_.injectPacket(noticePacket, PacketContainer::Direction::kServerToClient);
```