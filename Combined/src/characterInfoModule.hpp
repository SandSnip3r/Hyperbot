#ifndef CHARACTER_INFO_MODULE_HPP_
#define CHARACTER_INFO_MODULE_HPP_

#include "brokerSystem.hpp"
#include "packetParser.hpp"
#include "parsedPacket.hpp"
#include "shared/silkroad_security.h"

class CharacterInfoModule {
public:
  CharacterInfoModule(BrokerSystem &brokerSystem,
                      const packet::parsing::PacketParser &packetParser);
  bool handlePacket(const PacketContainer &packet);
private:
  BrokerSystem &broker_;
  const packet::parsing::PacketParser &packetParser_;
  void characterInfoReceived(const packet::parsing::ParsedServerAgentCharacterData &packet);
};

#endif // CHARACTER_INFO_MODULE_HPP_