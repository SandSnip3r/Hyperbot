#ifndef CHARACTER_INFO_MODULE_HPP_
#define CHARACTER_INFO_MODULE_HPP_

#include "brokerSystem.hpp"
#include "gameData.hpp"
#include "packetParser.hpp"
#include "parsedPacket.hpp"
#include "shared/silkroad_security.h"

#include <array>
#include <vector>

//TODO: Wrap in module namespace

enum class PotionType {
  kHp,
  kMp
};

enum class Race {
  kChinese,
  kEuropean
};

enum class Gender {
  kMale,
  kFemale
};

class CharacterInfoModule {
public:
  CharacterInfoModule(BrokerSystem &brokerSystem,
                      const packet::parsing::PacketParser &packetParser,
                      const pk2::media::GameData &gameData);
  bool handlePacket(const PacketContainer &packet);
private:
  static const int kEuPotionDefaultDelayMs{15000};
  static const int kChPotionDefaultDelayMs{1000};
  BrokerSystem &broker_;
  const packet::parsing::PacketParser &packetParser_;
  const pk2::media::GameData &gameData_;
  Race race_;
  Gender gender_;
  int hpPotionDelayMs_{1000};
  int mpPotionDelayMs_{1000};
  uint32_t uniqueId_{0};
  uint32_t hp_{0};
  uint32_t mp_{0};
  uint32_t maxHp_{1};
  uint32_t maxMp_{1};
  // Bitmask of all states (initialized as having no states)
  uint32_t prevStateBitmask_{0};
  // Set all states as level 0 (meaning there is no state)
  std::array<bool,6> legacyStates = {0};
  std::array<uint8_t,32> modernStateLevel = {0};

  std::map<uint8_t, item::Item*> inventoryItemMap_;
  void characterInfoReceived(const packet::parsing::ParsedServerAgentCharacterData &packet);
  void entityUpdateReceived(const packet::parsing::ParsedServerHpMpUpdate &packet);
  int getPotionDefaultDelay();
  void statUpdateReceived(const packet::parsing::ParsedServerAgentCharacterUpdateStats &packet);
  void setRaceAndGender(uint32_t refObjId);
  void serverUseItemReceived(const packet::parsing::ParsedServerUseItem &packet);
  void checkIfNeedToHeal();
  void usePotion(PotionType potionType);
  void useItem(uint8_t slotNum);
  void updateStates(uint32_t stateBitmask, const std::vector<uint8_t> &stateLevels);
};

#endif // CHARACTER_INFO_MODULE_HPP_