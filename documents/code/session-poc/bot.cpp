#include "bot.hpp"

#include <iostream>

bool Bot::handleServerCharacter(std::unique_ptr<PacketParsing::PacketParser> &packetParser) {
  PacketParsing::ServerCharacterPacket *serverCharacter = dynamic_cast<PacketParsing::ServerCharacterPacket*>(packetParser.get());
  if (serverCharacter == nullptr) {
    std::cerr << "Bot::handleClientChat Invalid PacketParser type\n";
  }
  maxHp_ = serverCharacter->maxHp();
  std::cout << "[Bot] Character MAX HP: " << maxHp_ << '\n';
}

bool Bot::handleServerHpMpUpdate(std::unique_ptr<PacketParsing::PacketParser> &packetParser) {
  const int kHpHealAmount{10};
  PacketParsing::ServerHpMpUpdatePacket *serverHpMpUpdate = dynamic_cast<PacketParsing::ServerHpMpUpdatePacket*>(packetParser.get());
  if (serverHpMpUpdate == nullptr) {
    std::cerr << "Bot::handleClientChat Invalid PacketParser type\n";
  }
  int currentHp = serverHpMpUpdate->hp();
  std::cout << "[Bot] Character HP update: " << currentHp << '\n';
  if (currentHp == 0) {
    std::cout << "[Bot] Character died\n";
    throw std::runtime_error("Simulation over");
  } else if (currentHp/static_cast<double>(maxHp_) < kMinimumHpPercentage) {
    // Hp too low
    std::cout << "[Bot] Character's health is too low! Use " << kHpHealAmount << "hp heal potion\n";
    Packet p(Packet::Opcode::kClientItemUse);
    p.setHpIncreaseAmount(kHpHealAmount);
    brokerSystem_.injectPacket(p, Packet::Direction::kBotToServer);
  }
}

Bot::Bot(BrokerSystem &brokerSystem) : brokerSystem_(brokerSystem) {
  // Register every interesting opcode to a handler
  brokerSystem_.subscribeToServerPacket(Packet::Opcode::kServerCharacter, std::bind(&Bot::handleServerCharacter, this, std::placeholders::_1));
  brokerSystem_.subscribeToServerPacket(Packet::Opcode::kServerHpMpUpdate, std::bind(&Bot::handleServerHpMpUpdate, this, std::placeholders::_1));
}