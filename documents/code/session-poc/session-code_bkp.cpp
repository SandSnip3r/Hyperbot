#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

class Packet {
public:
  enum class Direction { kClientToServer, kServerToClient, kBotToServer, kBotToClient };
  enum class Opcode { kClientItemMove, kClientPlayerAction, kClientStrUpdate, kClientChat };
private:
  Opcode opcode_;
public:
  Opcode opcode() const { return opcode_; }
};

namespace PacketParsing {

class PacketParser {
protected:
  bool parsed_{false};
  void parsedCheck() {
    if (!parsed_) {
      parsePacket();
      parsed_ = true;
    }
  }
  virtual void parsePacket() = 0;
};

class ClientChatPacket : public PacketParser {
private:
  const Packet &packet_;
  std::string message_;
  void parsePacket() override {
    // Parse packet
    message_ = "ClientChatPacket::parsePacket test";
  }
public:
  ClientChatPacket(const Packet &packet) : packet_(packet) {}
  const std::string& message() {
    parsedCheck();
    return message_;
  }
};

PacketParser* newPacketParser(const Packet &packet) {
  switch (packet.opcode()) {
    case Packet::Opcode::kClientChat:
      return new ClientChatPacket(packet);
    case Packet::Opcode::kClientItemMove:
      return nullptr;
    case Packet::Opcode::kClientPlayerAction:
      return nullptr;
    case Packet::Opcode::kClientStrUpdate:
      return nullptr;
  }
}

} // namespace PacketParsing

/// BrokerSystem
/// TODO: Maybe the PacketHandleFunctions should take a polymorphic type which is a wrapper around the packet that provides lazy evaluation for the data that's parsed
/// TODO: Unsubscription from opcodes
/// TODO: Allow registration of multiple events in one call if necessary
class BrokerSystem {
private:
  using PacketHandleFunction = function<bool(std::unique_ptr<PacketParsing::PacketParser>&)>;
  using PacketInjectionFunction = function<void(const Packet&, const Packet::Direction)>;
  using PacketSubscriptionMap = unordered_map<Packet::Opcode, vector<PacketHandleFunction>>;
  PacketSubscriptionMap clientPacketSubscriptions, serverPacketSubscriptions;
  PacketInjectionFunction injectionFunction_;
public:
  void setInjectionFunction(PacketInjectionFunction &&injectionFunction) {
    injectionFunction_ = std::move(injectionFunction);
  }

  bool packetReceived(const Packet &packet, Packet::Direction packetDirection) {
    // A new packet has arrived
    // First, determine which "event bus" to "put it on"
    PacketSubscriptionMap *subscriptionMap;
    if (packetDirection == Packet::Direction::kClientToServer) {
      // The packet goes on the Client->Server event bus
      subscriptionMap = &clientPacketSubscriptions;
    } else if (packetDirection == Packet::Direction::kServerToClient) {
      // The packet goes on the Server->Client event bus
      subscriptionMap = &serverPacketSubscriptions;
    }
    bool forwardPacket=true;
    // Check if anybody is subscribed to this packet
    auto subscriptionIt = subscriptionMap->find(packet.opcode());
    if (subscriptionIt != subscriptionMap->end()) {
      // We have one or more subscribers for this opcode, forward it to each
      vector<PacketHandleFunction> &handleFunctions = subscriptionIt->second;
      for (auto &handleFunction : handleFunctions) {
        // Wrap the packet into an object with lazy-eval-parsing
        std::unique_ptr<PacketParsing::PacketParser> packetParser{PacketParsing::newPacketParser(packet)};
        // Send this packet and make note if the packet should be forwarded
        forwardPacket &= handleFunction(packetParser);
      }
    }
    return forwardPacket;
  }

  void injectPacket(const Packet &packet, Packet::Direction packetDirection) {
    injectionFunction_(packet, packetDirection);
  }

  void subscribeToClientPacket(Packet::Opcode opcode, PacketHandleFunction &&handleFunc) {
    clientPacketSubscriptions.emplace(opcode, std::move(handleFunc));
  }

  void subscribeToServerPacket(Packet::Opcode opcode, PacketHandleFunction &&handleFunc) {
    serverPacketSubscriptions.emplace(opcode, std::move(handleFunc));
  }
};

/// Bot
/// This class is a starting point that will contain all logic
class Bot {
private:
  BrokerSystem brokerSystem_;
  void handleChatCommand(const std::string &command) {
    cout << "Handling chat command \"" << command << "\"\n";
  }

  bool handleClientChat(std::unique_ptr<PacketParsing::PacketParser> &packetParser) {
    const std::string kChatCommandPrefix{"\\b "};

    PacketParsing::ClientChatPacket *chatPacket = dynamic_cast<PacketParsing::ClientChatPacket*>(packetParser.get());
    if (chatPacket == nullptr) {
      std::cerr << "Bot::handleClientChat Invalid PacketParser type\n";
    }
    const std::string &msg = chatPacket->message();
    if (msg.find(kChatCommandPrefix) == 0) {
      // Message starts with \b, must be a chat command
      handleChatCommand(msg.substr(kChatCommandPrefix.size()));
    }
  }
public:
  Bot(BrokerSystem &brokerSystem) : brokerSystem_(brokerSystem) {
    // Register for certain events, for example, the STR update packet
    brokerSystem_.subscribeToClientPacket(Packet::Opcode::kClientChat, std::bind(&Bot::handleClientChat, this, std::placeholders::_1));
    // brokerSystem_.subscribeToClientPackets({Packet::Opcode::kClientStrUpdate}, std::bind(&Bot::handleClientPacket, this, std::placeholders::_1));
  }
};

class Proxy {
private:
  BrokerSystem broker_;
  void handlePacket(const Packet &packet, Packet::Direction packetDirection) {
    bool packetShouldBeForwarded = broker_.packetReceived(packet, packetDirection);
  }
public:
  Proxy(BrokerSystem &broker) : broker_(broker) {
    broker_.setInjectionFunction(std::bind(&Proxy::inject, this, std::placeholders::_1, std::placeholders::_2));
  }
  void inject(const Packet &packet, Packet::Direction packetDirection) {}
};

class Session {
public:
  BrokerSystem broker_;
  Proxy proxy_{broker_};
  Bot bot_{broker_};
  Session() {

  }
};

int main() {
  return 0;
}