#include "commonParsing.hpp"

#include <iostream>

namespace packet::parsing {

std::shared_ptr<storage::Item> parseGenericItem(StreamUtility &stream, const pk2::ItemData &itemData) {
  auto rentInfo = parseRentInfo(stream);

  uint32_t refItemId = stream.Read<uint32_t>();
  if (!itemData.haveItemWithId(refItemId)) {
    throw std::runtime_error("Unable to parse packet. Encountered an item (id:"+std::to_string(refItemId)+") for which we have no data on.");
  }
  const pk2::ref::Item &itemRef = itemData.getItemById(refItemId);
  std::shared_ptr<storage::Item> parsedItem{storage::newItemByTypeData(itemRef)};
  if (!parsedItem) {
    throw std::runtime_error("Unable to create an item object for item");
  }

  parseItem(parsedItem.get(), stream);
  return parsedItem;
}

structures::RentInfo parseRentInfo(StreamUtility &stream) {
  structures::RentInfo rentInfo;
  rentInfo.rentType = stream.Read<uint32_t>();
  
  if (rentInfo.rentType == 1) {
    rentInfo.canDelete = stream.Read<uint16_t>();
    rentInfo.periodBeginTime = stream.Read<uint32_t>();
    rentInfo.periodEndTime = stream.Read<uint32_t>();
  } else if (rentInfo.rentType == 2) {
    rentInfo.canDelete = stream.Read<uint16_t>();
    rentInfo.canRecharge = stream.Read<uint16_t>();
    rentInfo.meterRateTime = stream.Read<uint32_t>();
  } else if (rentInfo.rentType == 3) {
    rentInfo.canDelete = stream.Read<uint16_t>();
    rentInfo.canRecharge = stream.Read<uint16_t>();
    rentInfo.periodBeginTime = stream.Read<uint32_t>();
    rentInfo.periodEndTime = stream.Read<uint32_t>();
    rentInfo.packingTime = stream.Read<uint32_t>();
  }
  return rentInfo;
}

void parseItem(storage::ItemEquipment &item, StreamUtility &stream) {
  item.optLevel = stream.Read<uint8_t>();
  item.variance = stream.Read<uint64_t>();
  item.durability = stream.Read<uint32_t>();
  uint8_t magParamNum = stream.Read<uint8_t>();
  for (int paramIndex=0; paramIndex<magParamNum; ++paramIndex) {
    item.magicParams.emplace_back();
    auto &magParam = item.magicParams.back();
    magParam.type = stream.Read<uint32_t>();
    magParam.value = stream.Read<uint32_t>();
  }
  
  uint8_t bindingOptionType1 = stream.Read<uint8_t>(); // Weird useless byte (to represent Socket)
  uint8_t bindingOptionCount1 = stream.Read<uint8_t>();
  for (int bindingOptionIndex=0; bindingOptionIndex<bindingOptionCount1; ++bindingOptionIndex) {
    item.socketOptions.emplace_back();
    auto &socketOption = item.socketOptions.back();
    socketOption.slot = stream.Read<uint8_t>();
    socketOption.id = stream.Read<uint32_t>();
    socketOption.nParam1 = stream.Read<uint32_t>();
  }
  
  uint8_t bindingOptionType2 = stream.Read<uint8_t>(); // Weird useless byte (to represent Advanced Elixir)
  uint8_t bindingOptionCount2 = stream.Read<uint8_t>();
  for (int bindingOptionIndex=0; bindingOptionIndex<bindingOptionCount2; ++bindingOptionIndex) {
    item.advancedElixirOptions.emplace_back();
    auto &advancedElixirOption = item.advancedElixirOptions.back();
    advancedElixirOption.slot = stream.Read<uint8_t>();
    advancedElixirOption.id = stream.Read<uint32_t>();
    advancedElixirOption.optValue = stream.Read<uint32_t>();
  }
}

void parseItemCosSummoner(storage::ItemCosGrowthSummoner *cosSummoner, StreamUtility &stream) {  
  cosSummoner->lifeState = static_cast<storage::CosLifeState>(stream.Read<uint8_t>());
  if (cosSummoner->lifeState != storage::CosLifeState::kInactive) {
    cosSummoner->refObjID = stream.Read<uint32_t>();
    uint16_t nameLength = stream.Read<uint16_t>();
    cosSummoner->name = stream.Read_Ascii(nameLength);

    // Special case for ability pets
    if (cosSummoner->type == storage::ItemType::kItemCosAbilitySummoner) {
      auto *cosAbilitySummoner = dynamic_cast<storage::ItemCosAbilitySummoner*>(cosSummoner);
      if (cosAbilitySummoner != nullptr) {
        cosAbilitySummoner->secondsToRentEndTime = stream.Read<uint32_t>();
      } else {
        throw std::runtime_error("Trying to cast Item to type ItemCosAbilitySummoner failed");
      }
    }

    uint8_t timedJobCount = stream.Read<uint8_t>();
    for (int jobNum=0; jobNum<timedJobCount; ++jobNum) {
      cosSummoner->jobs.emplace_back();
      auto &job = cosSummoner->jobs.back();
      job.category = stream.Read<uint8_t>();
      job.jobId = stream.Read<uint32_t>();
      job.timeToKeep = stream.Read<uint32_t>();
      if (job.category == 5) {
        job.data1 = stream.Read<uint32_t>();
        job.data2 = stream.Read<uint8_t>();
      }
    }
  }
}

void parseItem(storage::ItemCosGrowthSummoner &item, StreamUtility &stream) {
  parseItemCosSummoner(&item, stream);
}

void parseItem(storage::ItemCosAbilitySummoner &item, StreamUtility &stream) {
  parseItemCosSummoner(&item, stream);
}

void parseItem(storage::ItemMonsterCapsule &item, StreamUtility &stream) {
  item.refObjID = stream.Read<uint32_t>();
}

void parseItem(storage::ItemStorage &item, StreamUtility &stream) {
  item.quantity = stream.Read<uint32_t>();
}

void parseItem(storage::ItemExpendable &item, StreamUtility &stream) {
  item.quantity = stream.Read<uint16_t>();
}

void parseItem(storage::ItemStone &item, StreamUtility &stream) {
  parseItem(*dynamic_cast<storage::ItemExpendable*>(&item), stream);

  item.attributeAssimilationProbability = stream.Read<uint8_t>();
}

void parseItem(storage::ItemMagicPop &item, StreamUtility &stream) {
  parseItem(*dynamic_cast<storage::ItemExpendable*>(&item), stream);

  uint8_t magParamCount = stream.Read<uint8_t>();
  for (int paramIndex=0; paramIndex<magParamCount; ++paramIndex) {
    item.magicParams.emplace_back();
    auto &magicParam = item.magicParams.back();
    magicParam.type = stream.Read<uint32_t>();
    magicParam.value = stream.Read<uint32_t>();
  }
}

void parseItem(storage::Item *item, StreamUtility &stream) {
  using namespace storage;
  if (item->type == ItemType::kItemEquipment) {
    auto *equipment = dynamic_cast<ItemEquipment*>(item);
    if (equipment != nullptr) {
      parseItem(*equipment, stream);
    } else {
      throw std::runtime_error("Trying to cast Item to type ItemEquipment failed");
    }
  } else if (item->type == ItemType::kItemCosAbilitySummoner) {
    auto *cosAbilitySummoner = dynamic_cast<ItemCosAbilitySummoner*>(item);
    if (cosAbilitySummoner != nullptr) {
      parseItem(*cosAbilitySummoner, stream);
    } else {
      throw std::runtime_error("Trying to cast Item to type ItemCosAbilitySummoner failed");
    }
  } else if (item->type == ItemType::kItemCosGrowthSummoner) {
    auto *cosGrowthSummoner = dynamic_cast<ItemCosGrowthSummoner*>(item);
    if (cosGrowthSummoner != nullptr) {
      parseItem(*cosGrowthSummoner, stream);
    } else {
      throw std::runtime_error("Trying to cast Item to type ItemCosGrowthSummoner failed");
    }
  } else if (item->type == ItemType::kItemMonsterCapsule) {
    auto *monsterCapsule = dynamic_cast<ItemMonsterCapsule*>(item);
    if (monsterCapsule != nullptr) {
      parseItem(*monsterCapsule, stream);
    } else {
      throw std::runtime_error("Trying to cast Item to type ItemMonsterCapsule failed");
    }
  } else if (item->type == ItemType::kItemStorage) {
    auto *storage = dynamic_cast<ItemStorage*>(item);
    if (storage != nullptr) {
      parseItem(*storage, stream);
    } else {
      throw std::runtime_error("Trying to cast Item to type ItemStorage failed");
    }
  } else if (item->type == ItemType::kItemStone) {
    auto *stone = dynamic_cast<ItemStone*>(item);
    if (stone != nullptr) {
      parseItem(*stone, stream);
    } else {
      throw std::runtime_error("Trying to cast Item to type ItemStone failed");
    }
  } else if (item->type == ItemType::kItemMagicPop) {
    auto *magicPop = dynamic_cast<ItemMagicPop*>(item);
    if (magicPop != nullptr) {
      parseItem(*magicPop, stream);
    } else {
      throw std::runtime_error("Trying to cast Item to type ItemMagicPop failed");
    }
  } else if (item->type == ItemType::kItemExpendable) {
    auto *expendable = dynamic_cast<ItemExpendable*>(item);
    if (expendable != nullptr) {
      parseItem(*expendable, stream);
    } else {
      throw std::runtime_error("Trying to cast Item to type ItemExpendable failed");
    }
  }
}

structures::SkillAction parseSkillAction(StreamUtility &stream) {
  structures::SkillAction result;
  result.actionFlag = stream.Read<uint8_t>();
  if (result.actionFlag & static_cast<uint8_t>(enums::ActionFlag::kAttack)) {
    uint8_t successiveHitCount = stream.Read<uint8_t>();
    uint8_t struckObjectCount = stream.Read<uint8_t>();
    for (int objNum=0; objNum<struckObjectCount; ++objNum) {
      structures::SkillActionHitObject hitObject;
      hitObject.objGlobalId = stream.Read<uint32_t>();
      for (int hitNum=0; hitNum<successiveHitCount; ++hitNum) {
        structures::SkillActionHitResult hit;
        hit.hitResult = static_cast<enums::HitResult>(stream.Read<uint8_t>());
        if (hit.hitResult == enums::HitResult::kBlocked || hit.hitResult == enums::HitResult::kCopy) {
          continue;
        }
        uint32_t damageData = stream.Read<uint32_t>();
        hit.damageFlag = static_cast<enums::DamageFlag>(damageData & 0xFF);
        hit.damage = (damageData >> 8);
        hit.effect = stream.Read<uint32_t>();

        if (hit.hitResult == enums::HitResult::kKnockdown || hit.hitResult == enums::HitResult::kKnockback) {
          // Only ever knocked down or knocked back, there is no combination
          hit.regionId = stream.Read<uint16_t>();
          uint32_t xAsInt = stream.Read<uint32_t>();
          hit.x = *reinterpret_cast<float*>(&xAsInt);
          uint32_t yAsInt = stream.Read<uint32_t>();
          hit.y = *reinterpret_cast<float*>(&yAsInt);
          uint32_t zAsInt = stream.Read<uint32_t>();
          hit.z = *reinterpret_cast<float*>(&zAsInt);
        } else if (static_cast<uint8_t>(hit.hitResult) == 7) {
          std::cout << "parseSkillAction: WHOAAAAA!!!! Unhandled skill end case. Unknown what this is!!!\n";
        }
        hitObject.hits.emplace_back(std::move(hit));
      }
      result.hitObjects.emplace_back(std::move(hitObject));
    }
  }
  if (result.actionFlag & static_cast<uint8_t>(enums::ActionFlag::kTeleport) || result.actionFlag & static_cast<uint8_t>(enums::ActionFlag::kSprint)) {
    result.regionId = stream.Read<uint16_t>();
    uint32_t xAsInt = stream.Read<uint32_t>();
    result.x = *reinterpret_cast<float*>(&xAsInt);
    uint32_t yAsInt = stream.Read<uint32_t>();
    result.y = *reinterpret_cast<float*>(&yAsInt);
    uint32_t zAsInt = stream.Read<uint32_t>();
    result.z = *reinterpret_cast<float*>(&zAsInt);
  }
  return result;
}

structures::Position parsePosition(StreamUtility &stream) {
  structures::Position position;
  position.regionId = stream.Read<uint16_t>();
  position.xOffset = stream.Read<float>();
  position.yOffset = stream.Read<float>();
  position.zOffset = stream.Read<float>();
  return position;
}

} // namespace packet::parsing