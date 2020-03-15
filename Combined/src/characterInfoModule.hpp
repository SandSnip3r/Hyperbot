#ifndef CHARACTER_INFO_MODULE_HPP_
#define CHARACTER_INFO_MODULE_HPP_

#include "brokerSystem.hpp"
#include "gameData.hpp"
#include "packetParser.hpp"
#include "parsedPacket.hpp"
#include "shared/silkroad_security.h"

#include <variant>
#include <vector>

//TODO: Wrap in module namespace

enum class PotionType {
  kHp,
  kMp
};

class CharacterInfoModule {
public:
  CharacterInfoModule(BrokerSystem &brokerSystem,
                      const packet::parsing::PacketParser &packetParser,
                      const pk2::media::GameData &gameData);
  bool handlePacket(const PacketContainer &packet);
private:
  BrokerSystem &broker_;
  const packet::parsing::PacketParser &packetParser_;
  const pk2::media::GameData &gameData_;
  uint32_t uniqueId_{0};
  uint32_t hp_{0};
  uint32_t mp_{0};
  uint32_t maxHp_{1};
  uint32_t maxMp_{1};
  std::map<uint8_t,packet::parsing::ParsedServerAgentCharacterData::ItemVariantType> inventoryItemMap_;
  void characterInfoReceived(const packet::parsing::ParsedServerAgentCharacterData &packet);
  void entityUpdateReceived(const packet::parsing::ParsedServerHpMpUpdate &packet);
  void statUpdateReceived(const packet::parsing::ParsedServerAgentCharacterUpdateStats &packet);
  void checkIfNeedToHeal();
  void usePotion(PotionType potionType);
};

#endif // CHARACTER_INFO_MODULE_HPP_