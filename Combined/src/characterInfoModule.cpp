#include "characterInfoModule.hpp"
#include "opcode.hpp"
#include "packetBuilding.hpp"

#include <iostream>
#include <memory>

CharacterInfoModule::CharacterInfoModule(BrokerSystem &brokerSystem,
                         const packet::parsing::PacketParser &packetParser,
                         const pk2::media::GameData &gameData) :
      broker_(brokerSystem),
      packetParser_(packetParser),
      gameData_(gameData) {
  auto packetHandleFunction = std::bind(&CharacterInfoModule::handlePacket, this, std::placeholders::_1);
  // Client packets
  // Server packets
  broker_.subscribeToServerPacket(Opcode::SERVER_CHARDATA, packetHandleFunction);
  broker_.subscribeToServerPacket(Opcode::SERVER_HPMP_UPDATE, packetHandleFunction);
  broker_.subscribeToServerPacket(Opcode::SERVER_STATS, packetHandleFunction);
  broker_.subscribeToServerPacket(Opcode::SERVER_ITEM_USE, packetHandleFunction);
}

bool CharacterInfoModule::handlePacket(const PacketContainer &packet) {
  std::unique_ptr<packet::parsing::ParsedPacket> parsedPacket;
  try {
    parsedPacket = packetParser_.parsePacket(packet);
  } catch (std::exception &ex) {
    std::cerr << "[CharacterInfoModule] Failed to parse packet " << std::hex << packet.opcode << std::dec << "\n  Error: \"" << ex.what() << "\"\n";
    return true;
  }

  if (!parsedPacket) {
    // Not yet parsing this packet
    return true;
  }

  packet::parsing::ParsedServerAgentCharacterData *charData = dynamic_cast<packet::parsing::ParsedServerAgentCharacterData*>(parsedPacket.get());
  if (charData != nullptr) {
    characterInfoReceived(*charData);
    return true;
  }

  packet::parsing::ParsedServerHpMpUpdate *hpMpUpdate = dynamic_cast<packet::parsing::ParsedServerHpMpUpdate*>(parsedPacket.get());
  if (hpMpUpdate != nullptr) {
    entityUpdateReceived(*hpMpUpdate);
    return true;
  }

  packet::parsing::ParsedServerUseItem *serverUseItemUpdate = dynamic_cast<packet::parsing::ParsedServerUseItem*>(parsedPacket.get());
  if (serverUseItemUpdate != nullptr) {
    serverUseItemReceived(*serverUseItemUpdate);
    return true;
  }

  packet::parsing::ParsedServerAgentCharacterUpdateStats *statUpdate = dynamic_cast<packet::parsing::ParsedServerAgentCharacterUpdateStats*>(parsedPacket.get());
  if (statUpdate != nullptr) {
    statUpdateReceived(*statUpdate);
    return true;
  }

  std::cout << "Unhandled packet subscribed to\n";
  return true;
}

void CharacterInfoModule::statUpdateReceived(const packet::parsing::ParsedServerAgentCharacterUpdateStats &packet) {
  std::cout << "Received stat update\n";
  maxHp_ = packet.maxHp();
  maxMp_ = packet.maxMp();
  checkIfNeedToHeal();
}

int CharacterInfoModule::getPotionDefaultDelay() {
  if (race_ == Race::kChinese) {
    return kChPotionDefaultDelayMs;
  } else {
    return kEuPotionDefaultDelayMs;
  }
}

void CharacterInfoModule::setRaceAndGender(uint32_t refObjId) {
  const auto &gameCharacterData = gameData_.characterData();
  if (!gameCharacterData.haveCharacterWithId(refObjId)) {
    std::cout << "Unable to determine race or gender. No \"item\" data for id: " << refObjId << '\n';
    return;
  }
  const auto &character = gameCharacterData.getCharacterById(refObjId);
  if (character.country == 0) {
    race_ = Race::kChinese;
  } else {
    race_ = Race::kEuropean;
  }
  hpPotionDelayMs_ = getPotionDefaultDelay();
  mpPotionDelayMs_ = getPotionDefaultDelay();
  if (character.charGender == 1) {
    gender_ = Gender::kMale;
  } else {
    gender_ = Gender::kFemale;
  }
}

void CharacterInfoModule::characterInfoReceived(const packet::parsing::ParsedServerAgentCharacterData &packet) {
  std::cout << "Character data received\n";
  auto refObjId = packet.refObjId();
  setRaceAndGender(refObjId);
  uniqueId_ = packet.entityUniqueId();
  hp_ = packet.hp();
  mp_ = packet.mp();
  inventoryItemMap_ = packet.inventoryItemMap();
  std::cout << "We are #" << uniqueId_ << ", and we have " << hp_ << " hp and " << mp_ << " mp\n";
  checkIfNeedToHeal();
}

void CharacterInfoModule::usePotion(PotionType potionType) {
  uint8_t typeId4;
  if (potionType == PotionType::kHp) {
    typeId4 = 1;
  } else if (potionType == PotionType::kMp) {
    typeId4 = 2;
  } else {
    std::cout << "CharacterInfoModule::usePotion: Potion type " << static_cast<int>(potionType) << " not supported\n";
    return;
  }
  // Find potion in inventory
  for (const auto &slotNumItemPair : inventoryItemMap_) {
    const auto &slotNum = slotNumItemPair.first;
    const item::Item *itemPtr = slotNumItemPair.second;
    const item::ItemExpendable *item;
    if ((item = dynamic_cast<const item::ItemExpendable*>(itemPtr)) != nullptr) {
      // Expendable item
      if (item->itemInfo->typeId1 == 3 && item->itemInfo->typeId2 == 3 && item->itemInfo->typeId3 == 1 && item->itemInfo->typeId4 == typeId4) {
        std::cout << "  We found a potion!! Trying to use it\n";
        useItem(slotNum);
        // Set a timeout for how long we must wait before retrying to use a potion
        break;
      }
    }
  }
}

void CharacterInfoModule::useItem(uint8_t slotNum) {
  // Note: We expect that the item is guaranteed to be in the inventory
  // Note: We expect no thread contention
  const item::Item *itemPtr = inventoryItemMap_.at(slotNum);
  uint16_t itemInfo = 0;
  itemInfo |= itemPtr->itemInfo->cashItem;
  itemInfo |= (itemPtr->itemInfo->bionic << 1);
  itemInfo |= (itemPtr->itemInfo->typeId1 << 2);
  itemInfo |= (itemPtr->itemInfo->typeId2 << 5);
  itemInfo |= (itemPtr->itemInfo->typeId3 << 7);
  itemInfo |= (itemPtr->itemInfo->typeId4 << 11);
  auto useItemPacket = PacketBuilding::ClientUseItemBuilder(slotNum, itemInfo).packet();
  broker_.injectPacket(useItemPacket, PacketContainer::Direction::kClientToServer);
}

void CharacterInfoModule::checkIfNeedToHeal() {
  if (maxHp_ == 1 || maxMp_ == 1) {
    // Dont yet know our max
    std::cout << "checkIfNeedToHeal: dont know max hp or mp\n";
    return;
  }
  if (maxHp_ == 0) {
    // Either uninitialized or dead. Cant heal in either case probably
    std::cout << "checkIfNeedToHeal: Either uninitialized or dead. Cant heal in either case probably\n";
    // TODO: Figure out
    return;
  }
  static const double kHpThreshold = 0.80;
  static const double kMpThreshold = 0.70;
  double hpPercentage = static_cast<double>(hp_)/maxHp_;
  double mpPercentage = static_cast<double>(mp_)/maxMp_;
  if (hpPercentage <= kHpThreshold) {
    std::cout << "Need to heal Hp (" << hpPercentage << "%)\n";
    usePotion(PotionType::kHp);
  }
  if (mpPercentage <= kMpThreshold) {
    std::cout << "Need to heal Mp (" << mpPercentage << "%)\n";
    usePotion(PotionType::kMp);
  }
}

std::string whyChanged(packet_enums::UpdateFlag updateFlag) {
  int resultCount = 0;
  std::string result;
  if (static_cast<int16_t>(updateFlag) & static_cast<int16_t>(packet_enums::UpdateFlag::kDamage)) {
    if (resultCount > 0) {
      result += ",";
    }
    result += "Damage";
    ++resultCount;
  }
  if (static_cast<int16_t>(updateFlag) & static_cast<int16_t>(packet_enums::UpdateFlag::kDotDamage)) {
    if (resultCount > 0) {
      result += ",";
    }
    result += "DotDamage";
    ++resultCount;
  }
  if (static_cast<int16_t>(updateFlag) & static_cast<int16_t>(packet_enums::UpdateFlag::kConsume)) {
    if (resultCount > 0) {
      result += ",";
    }
    result += "Consume";
    ++resultCount;
  }
  if (static_cast<int16_t>(updateFlag) & static_cast<int16_t>(packet_enums::UpdateFlag::kReverse)) {
    if (resultCount > 0) {
      result += ",";
    }
    result += "Reverse";
    ++resultCount;
  }
  if (static_cast<int16_t>(updateFlag) & static_cast<int16_t>(packet_enums::UpdateFlag::kRegeneration)) {
    if (resultCount > 0) {
      result += ",";
    }
    result += "Regeneration";
    ++resultCount;
  }
  if (static_cast<int16_t>(updateFlag) & static_cast<int16_t>(packet_enums::UpdateFlag::kPotion)) {
    if (resultCount > 0) {
      result += ",";
    }
    result += "Potion";
    ++resultCount;
  }
  if (static_cast<int16_t>(updateFlag) & static_cast<int16_t>(packet_enums::UpdateFlag::kHeal)) {
    if (resultCount > 0) {
      result += ",";
    }
    result += "Heal";
    ++resultCount;
  }
  if (static_cast<int16_t>(updateFlag) & static_cast<int16_t>(packet_enums::UpdateFlag::kUnknown128)) {
    if (resultCount > 0) {
      result += ",";
    }
    result += "Unknown128";
    ++resultCount;
  }
  if (static_cast<int16_t>(updateFlag) & static_cast<int16_t>(packet_enums::UpdateFlag::kAbnormalState)) {
    if (resultCount > 0) {
      result += ",";
    }
    result += "AbnormalState";
    ++resultCount;
  }
  if (static_cast<int16_t>(updateFlag) & static_cast<int16_t>(packet_enums::UpdateFlag::kMurderBurn)) {
    if (resultCount > 0) {
      result += ",";
    }
    result += "MurderBurn";
    ++resultCount;
  }
  return result;
}

void CharacterInfoModule::entityUpdateReceived(const packet::parsing::ParsedServerHpMpUpdate &packet) {
  std::cout << "Entity update received\n";
  if (packet.entityUniqueId() != uniqueId_) {
    // Not for my character, can ignore
    std::cout << "Not for me\n";
    return;
  }
  if (packet.vitalBitmask() & static_cast<uint8_t>(packet_enums::VitalInfoFlag::kVitalInfoHp)) {
    // Our HP changed
    if (hp_ != packet.newHpValue()) {
      std::cout << "Our HP changed! " << hp_ << " -> " << packet.newHpValue() << " (" << (static_cast<int64_t>(packet.newHpValue())-hp_) << ") because of " << whyChanged(packet.updateFlag()) << '\n';
      hp_ = packet.newHpValue();
    } else {
      std::cout << "Weird, says HP changed, but it didn't\n";
    }
  }
  if (packet.vitalBitmask() & static_cast<uint8_t>(packet_enums::VitalInfoFlag::kVitalInfoMp)) {
    // Our MP changed
    if (mp_ != packet.newMpValue()) {
      std::cout << "Our MP changed! " << mp_ << " -> " << packet.newMpValue() << " (" << (static_cast<int64_t>(packet.newMpValue())-mp_) << ") because of " << whyChanged(packet.updateFlag()) << '\n';
      mp_ = packet.newMpValue();
    } else {
      std::cout << "Weird, says MP changed, but it didn't\n";
    }
  }

  if (packet.vitalBitmask() & static_cast<uint8_t>(packet_enums::VitalInfoFlag::kVitalInfoAbnormal)) {
    // Our states changed
    auto stateBitmask = packet.stateBitmask();
    auto stateLevels = packet.stateLevels();
    updateStates(stateBitmask, stateLevels);
  }

  checkIfNeedToHeal();
  // uint32_t entityUniqueId() const;
  // packet_enums::UpdateFlag updateFlag() const;
  // uint8_t vitalBitmask() const;
  // uint32_t newHpValue() const;
  // uint32_t newMpValue() const;
  // uint16_t newHgpValue() const;
  // uint32_t stateBitmask() const;
  // const std::vector<uint8_t>& stateLevels() const;
}

std::string toStr(packet_enums::AbnormalStateFlag state) {
  if (state == packet_enums::AbnormalStateFlag::kNone) {
    return "none";
  } else if (state == packet_enums::AbnormalStateFlag::kFrozen) {
    return "frozen";
  } else if (state == packet_enums::AbnormalStateFlag::kFrostbitten) {
    return "frostbitten";
  } else if (state == packet_enums::AbnormalStateFlag::kShocked) {
    return "shocked";
  } else if (state == packet_enums::AbnormalStateFlag::kBurnt) {
    return "burnt";
  } else if (state == packet_enums::AbnormalStateFlag::kPoisoned) {
    return "poisoned";
  } else if (state == packet_enums::AbnormalStateFlag::kZombie) {
    return "zombie";
  } else if (state == packet_enums::AbnormalStateFlag::kSleep) {
    return "sleep";
  } else if (state == packet_enums::AbnormalStateFlag::kBind) {
    return "bind";
  } else if (state == packet_enums::AbnormalStateFlag::kDull) {
    return "dull";
  } else if (state == packet_enums::AbnormalStateFlag::kFear) {
    return "fear";
  } else if (state == packet_enums::AbnormalStateFlag::kShortSighted) {
    return "shortSighted";
  } else if (state == packet_enums::AbnormalStateFlag::kBleed) {
    return "bleed";
  } else if (state == packet_enums::AbnormalStateFlag::kPetrify) {
    return "petrify";
  } else if (state == packet_enums::AbnormalStateFlag::kDarkness) {
    return "darkness";
  } else if (state == packet_enums::AbnormalStateFlag::kStunned) {
    return "stunned";
  } else if (state == packet_enums::AbnormalStateFlag::kDisease) {
    return "disease";
  } else if (state == packet_enums::AbnormalStateFlag::kConfusion) {
    return "confusion";
  } else if (state == packet_enums::AbnormalStateFlag::kDecay) {
    return "decay";
  } else if (state == packet_enums::AbnormalStateFlag::kWeak) {
    return "weak";
  } else if (state == packet_enums::AbnormalStateFlag::kImpotent) {
    return "impotent";
  } else if (state == packet_enums::AbnormalStateFlag::kDivision) {
    return "division";
  } else if (state == packet_enums::AbnormalStateFlag::kPanic) {
    return "panic";
  } else if (state == packet_enums::AbnormalStateFlag::kCombustion) {
    return "combustion";
  } else if (state == packet_enums::AbnormalStateFlag::kEmptyBit23) {
    return "emptyBit23";
  } else if (state == packet_enums::AbnormalStateFlag::kHidden) {
    return "hidden";
  } else if (state == packet_enums::AbnormalStateFlag::kEmptyBit25) {
    return "emptyBit25";
  } else if (state == packet_enums::AbnormalStateFlag::kEmptyBit26) {
    return "emptyBit26";
  } else if (state == packet_enums::AbnormalStateFlag::kEmptyBit27) {
    return "emptyBit27";
  } else if (state == packet_enums::AbnormalStateFlag::kEmptyBit28) {
    return "emptyBit28";
  } else if (state == packet_enums::AbnormalStateFlag::kEmptyBit29) {
    return "emptyBit29";
  } else if (state == packet_enums::AbnormalStateFlag::kEmptyBit30) {
    return "emptyBit30";
  } else if (state == packet_enums::AbnormalStateFlag::kEmptyBit31) {
    return "emptyBit31";
  }
}

void CharacterInfoModule::updateStates(uint32_t stateBitmask, const std::vector<uint8_t> &stateLevels) {
  uint32_t newlyReceivedStates = (prevStateBitmask_ ^ stateBitmask) & stateBitmask;
  uint32_t expiredStates = (prevStateBitmask_ ^ stateBitmask) & prevStateBitmask_;
  prevStateBitmask_ = stateBitmask;

  int stateLevelIndex=0;
  if (newlyReceivedStates != 0) {
    // We have some new states!
    for (int bitNum=0; bitNum<32; ++bitNum) {
      const auto kBit = static_cast<uint32_t>(1) << bitNum;
      if ((newlyReceivedStates & kBit) != 0) {
        const auto kState = static_cast<packet_enums::AbnormalStateFlag>(kBit);
        if (kState <= packet_enums::AbnormalStateFlag::kZombie) {
          // Legacy state
          legacyStates[bitNum] = true;
          std::cout << "We now are " << toStr(kState) << "\n";
        } else {
          // Modern state
          modernStateLevel[bitNum] = stateLevels[stateLevelIndex];
          ++stateLevelIndex;
          std::cout << "We now are under " << toStr(kState) << "\n";
          if (kState == packet_enums::AbnormalStateFlag::kPanic) {
            // HP Potions have a 4 second longer delay
            hpPotionDelayMs_ = getPotionDefaultDelay() + 4000;
            std::cout << "hp potion delay updated to " << hpPotionDelayMs_ << '\n';
          } else if (kState == packet_enums::AbnormalStateFlag::kCombustion) {
            // MP Potions have a 4 second longer delay
            mpPotionDelayMs_ = getPotionDefaultDelay() + 4000;
            std::cout << "mp potion delay updated to " << mpPotionDelayMs_ << '\n';
          }
        }
      }
    }
  }
  if (expiredStates != 0) {
    // We have some expired states
    for (int bitNum=0; bitNum<32; ++bitNum) {
      const auto kBit = static_cast<uint32_t>(1) << bitNum;
      if ((expiredStates & kBit) != 0) {
        const auto kState = static_cast<packet_enums::AbnormalStateFlag>(kBit);
        if (kState <= packet_enums::AbnormalStateFlag::kZombie) {
          // Legacy state
          legacyStates[bitNum] = false;
          std::cout << "We are no longer " << toStr(kState) << "\n";
        } else {
          // Modern state
          modernStateLevel[bitNum] = 0;
          std::cout << "We are no longer under " << toStr(kState) << "\n";
          if (kState == packet_enums::AbnormalStateFlag::kPanic) {
            // Restore potion delay to normal
            hpPotionDelayMs_ = getPotionDefaultDelay();
            std::cout << "hp potion delay updated to " << hpPotionDelayMs_ << '\n';
          } else if (kState == packet_enums::AbnormalStateFlag::kCombustion) {
            // Restore potion delay to normal
            mpPotionDelayMs_ = getPotionDefaultDelay();
            std::cout << "mp potion delay updated to " << mpPotionDelayMs_ << '\n';
          }
        }
      }
    }
  }
}

void CharacterInfoModule::serverUseItemReceived(const packet::parsing::ParsedServerUseItem &packet) {
}
