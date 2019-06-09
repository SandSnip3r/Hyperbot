#include "brokerSystem.hpp"

#ifndef BOT_HPP
#define BOT_HPP

/// Bot
/// This class is a starting point that will contain all logic
class Bot {
private:
  BrokerSystem &brokerSystem_;
  int maxHp_;
  const double kMinimumHpPercentage{0.7};
  bool handleServerCharacter(std::unique_ptr<PacketParsing::PacketParser> &packetParser);
  bool handleServerHpMpUpdate(std::unique_ptr<PacketParsing::PacketParser> &packetParser);
public:
  Bot(BrokerSystem &brokerSystem);
};

#endif // BOT_HPP