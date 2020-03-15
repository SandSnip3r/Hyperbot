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

void CharacterInfoModule::characterInfoReceived(const packet::parsing::ParsedServerAgentCharacterData &packet) {
  std::cout << "Character data received\n";
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
    const auto &item = slotNumItemPair.second;
    if (std::holds_alternative<item::ItemExpendable>(item)) {
      // Expendable item
      const auto &specificItem = std::get<item::ItemExpendable>(item);
      if (specificItem.itemInfo->typeId1 == 3 && specificItem.itemInfo->typeId2 == 3 && specificItem.itemInfo->typeId3 == 1 && specificItem.itemInfo->typeId4 == typeId4) {
        std::cout << "  We found a potion!!\n";
        uint16_t itemInfo = 0;
        itemInfo |= specificItem.itemInfo->cashItem;
        itemInfo |= (specificItem.itemInfo->bionic << 1);
        itemInfo |= (specificItem.itemInfo->typeId1 << 2);
        itemInfo |= (specificItem.itemInfo->typeId2 << 5);
        itemInfo |= (specificItem.itemInfo->typeId3 << 7);
        itemInfo |= (specificItem.itemInfo->typeId4 << 11);
        auto useItemPacket = PacketBuilding::ClientUseItemBuilder(slotNum, itemInfo).packet();
        broker_.injectPacket(useItemPacket, PacketContainer::Direction::kClientToServer);
        break;
      }
    }
  }
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