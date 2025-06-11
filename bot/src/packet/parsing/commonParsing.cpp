#include "commonParsing.hpp"
#include "entity/character.hpp"
#include "entity/entity.hpp"
#include "entity/item.hpp"
#include "entity/monster.hpp"
#include "entity/nonplayerCharacter.hpp"
#include "entity/playerCharacter.hpp"
#include "entity/portal.hpp"
#include "type_id/categories.hpp"
#include "type_id/typeCategory.hpp"

#include <silkroad_lib/position_math.hpp>
#include <silkroad_lib/scalar_types.hpp>
#include <silkroad_lib/pk2/ref/item.hpp>

#include <absl/log/log.h>

namespace packet::parsing {

namespace {

std::shared_ptr<entity::Entity> newObjectFromId(sro::scalar_types::ReferenceObjectId refObjId, const sro::pk2::CharacterData &characterData, const sro::pk2::ItemData &itemData, const sro::pk2::TeleportData &teleportData) {
  if (characterData.haveCharacterWithId(refObjId) &&
      characterData.getCharacterById(refObjId).typeId1 == 1) {
    const auto &character = characterData.getCharacterById(refObjId);
    if (character.typeId2 == 1) {
      auto ptr = std::make_shared<entity::PlayerCharacter>();
      ptr->refObjId = refObjId;
      ptr->typeId1 = character.typeId1;
      ptr->typeId2 = character.typeId2;
      ptr->typeId3 = character.typeId3;
      ptr->typeId4 = character.typeId4;
      return ptr;
    } else if (character.typeId2 == 2 && character.typeId3 == 1) {
      auto ptr = std::make_shared<entity::Monster>();
      ptr->refObjId = refObjId;
      ptr->typeId1 = character.typeId1;
      ptr->typeId2 = character.typeId2;
      ptr->typeId3 = character.typeId3;
      ptr->typeId4 = character.typeId4;
      return ptr;
    } else {
      auto ptr = std::make_shared<entity::NonplayerCharacter>();
      ptr->refObjId = refObjId;
      ptr->typeId1 = character.typeId1;
      ptr->typeId2 = character.typeId2;
      ptr->typeId3 = character.typeId3;
      ptr->typeId4 = character.typeId4;
      return ptr;
    }
  } else if (itemData.haveItemWithId(refObjId) && itemData.getItemById(refObjId).typeId1 == 3) {
    const auto &item = itemData.getItemById(refObjId);
    auto ptr = std::make_shared<entity::Item>();
    ptr->refObjId = refObjId;
    ptr->typeId1 = item.typeId1;
    ptr->typeId2 = item.typeId2;
    ptr->typeId3 = item.typeId3;
    ptr->typeId4 = item.typeId4;
    return ptr;
  } else if (teleportData.haveTeleportWithId(refObjId) && teleportData.getTeleportById(refObjId).typeId1 == 4) {
    const auto &portal = teleportData.getTeleportById(refObjId);
    auto ptr = std::make_shared<entity::Portal>();
    ptr->refObjId = refObjId;
    ptr->typeId1 = portal.typeId1;
    ptr->typeId2 = portal.typeId2;
    ptr->typeId3 = portal.typeId3;
    ptr->typeId4 = portal.typeId4;
    return ptr;
  }
  throw std::runtime_error("Do not know how to create this entity");
}

} // anonymous namespace

std::shared_ptr<storage::Item> parseGenericItem(StreamUtility &stream, const sro::pk2::ItemData &itemData) {
  auto rentInfo = parseRentInfo(stream);

  uint32_t refItemId = stream.Read<uint32_t>();
  if (!itemData.haveItemWithId(refItemId)) {
    throw std::runtime_error("Unable to parse packet. Encountered an item (id:"+std::to_string(refItemId)+") for which we have no data on.");
  }
  const sro::pk2::ref::Item &itemRef = itemData.getItemById(refItemId);
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
  auto parseInt32Pos = [](auto &stream) {
    const auto regionId = stream.template Read<sro::RegionId>();
    const auto xOffset = stream.template Read<int32_t>();
    const auto yOffset = stream.template Read<int32_t>();
    const auto zOffset = stream.template Read<int32_t>();
    return sro::Position(regionId, xOffset, yOffset, zOffset);
  };
  structures::SkillAction skillAction;
  skillAction.actionFlag = stream.Read<enums::ActionFlag>();
  if (flags::isSet(skillAction.actionFlag, enums::ActionFlag::kAttack)) {
    uint8_t successiveHitCount = stream.Read<uint8_t>();
    uint8_t struckObjectCount = stream.Read<uint8_t>();
    for (int objNum=0; objNum<struckObjectCount; ++objNum) {
      structures::SkillActionHitObject hitObject;
      hitObject.targetGlobalId = stream.Read<sro::scalar_types::EntityGlobalId>();
      for (int hitNum=0; hitNum<successiveHitCount; ++hitNum) {
        structures::SkillActionHitResult hit;
        hit.hitResultFlag = static_cast<enums::HitResult>(stream.Read<uint8_t>());
        if (static_cast<uint8_t>(hit.hitResultFlag) & static_cast<uint8_t>(enums::HitResult::kBlocked) || static_cast<uint8_t>(hit.hitResultFlag) & static_cast<uint8_t>(enums::HitResult::kCopy)) {
          continue;
        }
        uint32_t damageData = stream.Read<uint32_t>();
        hit.damageFlag = static_cast<enums::DamageFlag>(damageData & 0xFF);
        hit.damage = (damageData >> 8);
        hit.effect = stream.Read<uint32_t>();

        if (static_cast<uint8_t>(hit.hitResultFlag) & static_cast<uint8_t>(enums::HitResult::kKnockdown) || static_cast<uint8_t>(hit.hitResultFlag) & static_cast<uint8_t>(enums::HitResult::kKnockback)) {
          // Only ever knocked down or knocked back, there is no combination
          hit.position = parseInt32Pos(stream);
        } else if (static_cast<uint8_t>(hit.hitResultFlag) == 7) {
          LOG(WARNING) << "parseSkillAction: WHOAAAAA!!!! Unhandled skill end case. Unknown what this is!!!";
        }
        hitObject.hits.emplace_back(std::move(hit));
      }
      skillAction.hitObjects.emplace_back(std::move(hitObject));
    }
  }
  if (flags::isSet(skillAction.actionFlag, enums::ActionFlag::kTeleport) || flags::isSet(skillAction.actionFlag, enums::ActionFlag::kSprint)) {
    skillAction.position = parseInt32Pos(stream);
  }
  return skillAction;
}

sro::Position parsePosition(StreamUtility &stream) {
  sro::RegionId regionId = stream.Read<sro::RegionId>();
  float xOffset = stream.Read<float>();
  float yOffset = stream.Read<float>();
  float zOffset = stream.Read<float>();
  return sro::Position(regionId, xOffset, yOffset, zOffset);
}

std::shared_ptr<entity::Entity> parseSpawn(StreamUtility &stream,
                                           const sro::pk2::CharacterData &characterData,
                                           const sro::pk2::ItemData &itemData,
                                           const sro::pk2::SkillData &skillData,
                                           const sro::pk2::TeleportData &teleportData) {
  using namespace type_id;
  const auto refObjId = stream.Read<sro::scalar_types::ReferenceObjectId>();
  if (refObjId == std::numeric_limits<sro::scalar_types::ReferenceObjectId>::max()) {
    // Special case, refObjId == -1
    LOG(INFO) << "EVENT_ZONE";
    // EVENT_ZONE (Traps, Buffzones, ...)
    uint16_t eventZoneTypeId = stream.Read<uint16_t>();
    LOG(INFO) << " eventZoneTypeId:" << eventZoneTypeId;
    sro::scalar_types::ReferenceObjectId eventZoneRefSkillId = stream.Read<sro::scalar_types::ReferenceObjectId>();
    LOG(INFO) << " eventZoneRefSkillId:" << eventZoneRefSkillId;
    sro::scalar_types::EntityGlobalId globalId = stream.Read<sro::scalar_types::EntityGlobalId>();
    LOG(INFO) << " globalId:" << globalId;
    const auto position = parsePosition(stream);
    LOG(INFO) << " position:" << position;
    sro::Angle angle = stream.Read<sro::Angle>();
    LOG(INFO) << " angle:" << angle;
    return {};
  }

  std::shared_ptr<entity::Entity> entity = newObjectFromId(refObjId, characterData, itemData, teleportData);
  if (!entity) {
    throw std::runtime_error("Failed to create object for spawn");
  }

  if (characterData.haveCharacterWithId(entity->refObjId)) {
    const auto &character = characterData.getCharacterById(entity->refObjId);
    const auto characterTypeId = type_id::getTypeId(character);
    if (!categories::kCharacter.contains(characterTypeId)) {
      throw std::runtime_error("Have a character, but type id does not match");
    }
    entity::Character *characterPtr = dynamic_cast<entity::Character*>(entity.get());
    if (characterPtr == nullptr) {
      throw std::runtime_error("parseSpawn, have a character, but the entity pointer cannot be cast to a Character");
    }
    bool characterHasJobEquipmentInInventory = false;
    if (categories::kPlayerCharacter.contains(characterTypeId)) {
      entity::PlayerCharacter *playerCharacterPtr = dynamic_cast<entity::PlayerCharacter*>(entity.get());
      if (playerCharacterPtr == nullptr) {
        throw std::runtime_error("parseSpawn, have a player character, but the entity pointer cannot be cast to a PlayerCharacter");
      }
      uint8_t scale = stream.Read<uint8_t>();
      uint8_t hwanLevel = stream.Read<uint8_t>();
      stream.Read(playerCharacterPtr->freePvpMode);
      uint8_t autoInverstExp = stream.Read<uint8_t>();  //1 = Beginner Icon, 2 = Helpful, 3 = Beginner & Helpful

      // Inventory
      uint8_t inventorySize = stream.Read<uint8_t>();
      uint8_t inventoryItemCount = stream.Read<uint8_t>();
      for (int i=0; i<inventoryItemCount; ++i) {
        uint32_t itemRefId = stream.Read<uint32_t>();
        if (!itemData.haveItemWithId(itemRefId)) {
          throw std::runtime_error("Parsing ServerAgentGroupSpawn, found item in character's inventory which we have no data on");
        }
        const auto &item = itemData.getItemById(itemRefId);
        const auto itemTypeId = type_id::getTypeId(item);
        if (categories::kEquipment.contains(itemTypeId)) {
          uint8_t optLevel = stream.Read<uint8_t>();
        }
        if (categories::kTraderSuit.contains(itemTypeId) ||
            categories::kThiefSuit.contains(itemTypeId) ||
            categories::kHunterSuit.contains(itemTypeId)) {
          characterHasJobEquipmentInInventory = true;
        }
      }

      // AvatarInventory
      uint8_t avatarInventorySize = stream.Read<uint8_t>();
      uint8_t avatarInventoryItemCount = stream.Read<uint8_t>();
      for (int i=0; i<avatarInventoryItemCount; ++i) {
        uint32_t itemRefId = stream.Read<uint32_t>();
        if (!itemData.haveItemWithId(itemRefId)) {
          throw std::runtime_error("Parsing ServerAgentGroupSpawn, found item in character's avatar inventory which we have no data on");
        }
        const auto &item = itemData.getItemById(itemRefId);
        const auto itemTypeId = type_id::getTypeId(item);
        if (categories::kEquipment.contains(itemTypeId)) {
          uint8_t optLevel = stream.Read<uint8_t>();
        }
      }

      // Mask
      bool hasMask = stream.Read<uint8_t>();
      if (hasMask) {
        uint32_t maskRefObjId = stream.Read<uint32_t>();
        if (!itemData.haveItemWithId(maskRefObjId)) {
          throw std::runtime_error("Parsing ServerAgentGroupSpawn, found mask on character which we have no data on");
        }
        const auto &maskItem = itemData.getItemById(maskRefObjId);
        if (maskItem.typeId1 == character.typeId1 &&
            maskItem.typeId2 == character.typeId2) {
          // Duplicate
          uint8_t maskScale = stream.Read<uint8_t>();
          uint8_t maskItemCount = stream.Read<uint8_t>();
          for (int i=0; i<maskItemCount; ++i) {
            uint32_t maskItemRefId = stream.Read<uint32_t>();
          }
        }
      }
    } else if (categories::kSiegeStruct.contains(characterTypeId)) {
      uint32_t structureHp = stream.Read<uint32_t>();
      uint32_t structureRefEventStructId = stream.Read<uint32_t>();
      uint16_t structureState = stream.Read<uint16_t>();
    }

    characterPtr->globalId = stream.Read<uint32_t>();

    // Position
    characterPtr->initializePosition(parsePosition(stream));
    sro::Angle angle = stream.Read<sro::Angle>();
    characterPtr->initializeAngle(angle);

    bool movementHasDestination = stream.Read<uint8_t>();

    uint8_t movementType = stream.Read<uint8_t>(); // 0 = walk, 1 = run
    characterPtr->lastMotionState = (movementType == 0 ? entity::MotionState::kWalk : entity::MotionState::kRun);

    if (movementHasDestination) {
      // Mouse destination
      auto destinationRegionId = stream.Read<sro::RegionId>();
      float offsetX, offsetY, offsetZ;
      if (sro::position_math::regionIsDungeon(destinationRegionId)) {
        // Dungeon
        offsetX = stream.Read<int32_t>();
        offsetY = stream.Read<int32_t>();
        offsetZ = stream.Read<int32_t>();
      } else {
        // World
        offsetX = stream.Read<int16_t>();
        offsetY = stream.Read<int16_t>();
        offsetZ = stream.Read<int16_t>();
      }
      if (std::trunc(characterPtr->position().xOffset()) == offsetX &&
          std::trunc(characterPtr->position().zOffset()) == offsetZ) {
        // Entity is not actually moving
      } else {
        // Entity is currently moving
        characterPtr->initializeAsMoving({destinationRegionId, offsetX, offsetY, offsetZ});
      }
    } else {
      packet::enums::AngleAction angleAction_ = static_cast<packet::enums::AngleAction>(stream.Read<uint8_t>());
      sro::Angle angle = stream.Read<sro::Angle>(); // Represents the new angle, character is looking at
      // For monsters, I think this means that they have never moved before, otherwise, they will have a destination that is the last point they moved to
      if (angleAction_ == packet::enums::AngleAction::kGoForward) {
        // Entity is currently moving
        characterPtr->initializeAsMoving(angle);
      }
    }

    // State
    characterPtr->lifeState = stream.Read<sro::entity::LifeState>();
    uint8_t unkByte0 = stream.Read<uint8_t>(); // Obsolete
    characterPtr->motionState = stream.Read<entity::MotionState>();
    if (characterPtr->motionState == entity::MotionState::kRun || characterPtr->motionState == entity::MotionState::kWalk) {
      // Save whether we were walking or running last
      // TODO: This should be done in a function in the MobileEntity
      characterPtr->lastMotionState = characterPtr->motionState;
    }
    uint8_t bodyState = stream.Read<uint8_t>(); // 0=None, 1=Hwan, 2=Untouchable, 3=GameMasterInvincible, 5=GameMasterInvisible, 6=Stealth, 7=Invisible
    characterPtr->walkSpeed = stream.Read<float>();
    characterPtr->runSpeed = stream.Read<float>();
    float hwanSpeed = stream.Read<float>();

    // Buffs
    uint8_t buffCount = stream.Read<uint8_t>();
    for (int i=0; i<buffCount; ++i) {
      sro::scalar_types::ReferenceSkillId skillRefId = stream.Read<sro::scalar_types::ReferenceSkillId>();
      sro::scalar_types::BuffTokenType token = stream.Read<sro::scalar_types::BuffTokenType>();
      characterPtr->addBuff(skillRefId, token, std::nullopt);
      if (!skillData.haveSkillWithId(skillRefId)) {
        throw std::runtime_error(absl::StrFormat("Parsing ServerAgentGroupSpawn, found buff (%d) which we have no data on", skillRefId));
      }
      const auto &skill = skillData.getSkillById(skillRefId);
      if (skill.isEfta()) {
        uint8_t creatorFlag = stream.Read<uint8_t>(); // 1=Creator, 2=Other
      }
    }

    if (categories::kPlayerCharacter.contains(characterTypeId)) {
      entity::PlayerCharacter *playerCharacterPtr = dynamic_cast<entity::PlayerCharacter*>(entity.get());
      if (playerCharacterPtr == nullptr) {
        throw std::runtime_error("parseSpawn, have a player character, but the entity pointer cannot be cast to a PlayerCharacter");
      }
      uint16_t nameLength = stream.Read<uint16_t>();
      playerCharacterPtr->name = stream.Read_Ascii(nameLength);

      uint8_t jobType = stream.Read<uint8_t>(); // 0=None, 1=Trader, 2=Thief, 3=Hunter
      uint8_t jobLevel = stream.Read<uint8_t>();
      uint8_t murderState = stream.Read<uint8_t>(); //0=White (Neutral), 1=Purple (Assaulter), 2=Red (Murder)
      bool isRiding = stream.Read<uint8_t>();
      bool inCombat = stream.Read<uint8_t>();
      if(isRiding) {
        uint32_t transportUniqueId = stream.Read<uint32_t>();
      }
      uint8_t scrollMode = stream.Read<uint8_t>(); // 0=None, 1=Return Scroll, 2=Bandit Return Scroll
      uint8_t interactMode = stream.Read<uint8_t>(); //0=None 2=P2P, 4=P2N_TALK, 6=OPNMKT_DEAL
      // exchange, talking to npc, stalling, etc
      uint8_t unkByte4 = stream.Read<uint8_t>();

      //Guild
      uint16_t guildNameLength = stream.Read<uint16_t>();
      std::string guildName = stream.Read_Ascii(guildNameLength);

      if (!characterHasJobEquipmentInInventory) {
        uint32_t guildId = stream.Read<uint32_t>();
        uint16_t guildMemberNicknameLength = stream.Read<uint16_t>();
        std::string guildMemberNickname = stream.Read_Ascii(guildMemberNicknameLength);
        uint32_t guildLastCrestRev = stream.Read<uint32_t>();
        uint32_t unionId = stream.Read<uint32_t>();
        uint32_t unionLastCrestRev = stream.Read<uint32_t>();
        uint8_t guildIsFriendly = stream.Read<uint8_t>(); // 0 = Hostile, 1 = Friendly
        uint8_t guildMemberSiegeAuthority = stream.Read<uint8_t>(); // See SiegeAuthority.cs
      }

      if (interactMode == 4) {
        uint16_t stallNameLength = stream.Read<uint16_t>();
        std::string stallName = stream.Read_Ascii(stallNameLength);
        uint32_t stallDecorationRefId = stream.Read<uint32_t>();
      }

      uint8_t equipmentCooldown = stream.Read<uint8_t>(); // Yellow bar when equipping or unequipping
      uint8_t pkFlag = stream.Read<uint8_t>();
    } else if (categories::kNonPlayerCharacter.contains(characterTypeId)) {
      uint8_t talkFlag = stream.Read<uint8_t>();
      if (talkFlag == 2) {
        uint8_t talkOptionCount = stream.Read<uint8_t>();
        std::vector<uint8_t> talkOptions;
        for (int i=0; i<talkOptionCount; ++i) {
          talkOptions.emplace_back(stream.Read<uint8_t>());
        }
      }

      if (categories::kMonster.contains(characterTypeId)) {
        entity::Monster *monsterPtr = dynamic_cast<entity::Monster*>(entity.get());
        if (monsterPtr == nullptr) {
          throw std::runtime_error("parseSpawn, have a monster, but the entity pointer cannot be cast to a Monster");
        }
        monsterPtr->rarity = stream.Read<sro::entity::MonsterRarity>();
        if (categories::kThiefMonster.contains(characterTypeId) ||
            categories::kHunterMonster.contains(characterTypeId)) {
          uint8_t appearance = stream.Read<uint8_t>();
        }
      } else if (categories::kCos.contains(characterTypeId)) {
        if (categories::kTransport.contains(characterTypeId) ||
            categories::kSilkPet.contains(characterTypeId) ||
            categories::kGoldPet.contains(characterTypeId) ||
            categories::kMercenary.contains(characterTypeId) ||
            categories::kCaptured.contains(characterTypeId) ||
            categories::kFollower.contains(characterTypeId) ||
            categories::kAssaulter.contains(characterTypeId)) {
          if (categories::kSilkPet.contains(characterTypeId) ||
              categories::kGoldPet.contains(characterTypeId)) {
            uint16_t nameLength = stream.Read<uint16_t>();
            std::string name = stream.Read_Ascii(nameLength);
          } else if (categories::kMercenary.contains(characterTypeId)) {
            uint16_t guildNameLength = stream.Read<uint16_t>();
            std::string guildName = stream.Read_Ascii(guildNameLength);
          }

          if (categories::kTransport.contains(characterTypeId) ||
              categories::kSilkPet.contains(characterTypeId) ||
              categories::kGoldPet.contains(characterTypeId) ||
              categories::kMercenary.contains(characterTypeId) ||
              categories::kCaptured.contains(characterTypeId)) {
            uint16_t ownerNameLength = stream.Read<uint16_t>();
            std::string ownerName = stream.Read_Ascii(ownerNameLength);

            if (categories::kTransport.contains(characterTypeId) ||
                categories::kSilkPet.contains(characterTypeId) ||
                categories::kGoldPet.contains(characterTypeId) ||
                categories::kMercenary.contains(characterTypeId)) {
              uint8_t ownerJobType = stream.Read<uint8_t>();

              if (categories::kTransport.contains(characterTypeId) ||
                  categories::kSilkPet.contains(characterTypeId) ||
                  categories::kMercenary.contains(characterTypeId)) {
                uint8_t ownerPvpState = stream.Read<uint8_t>();
                if (categories::kMercenary.contains(characterTypeId)) {
                  uint32_t ownerRefId = stream.Read<uint32_t>();
                }
              }
            }
          }
          uint32_t ownerUniqueId = stream.Read<uint32_t>();
        }
      } else if (categories::kSiegeObject.contains(characterTypeId)) {
        uint32_t guildId = stream.Read<uint32_t>();
        uint16_t guildNameLength = stream.Read<uint16_t>();
        std::string guildName = stream.Read_Ascii(guildNameLength);
      } else if (categories::kSiegeStruct.contains(characterTypeId)) {
        LOG(INFO) << "CGObjSiegeStruct encountered in parseSpawn, but unhandled";
      //   // CGObjSiegeStruct
      //   uint32_t unk0 = stream.Read<uint32_t>(); // 0xFFFFFFFF
      //   uint16_t unk1 = stream.Read<uint16_t>(); // 0x0054
      //   uint32_t unk2 = stream.Read<uint32_t>(); // 0x000052FE
      //   uint32_t unk3 = stream.Read<uint32_t>(); // 0x0001ACB9
      //   uint16_t unk4 = stream.Read<uint16_t>(); // 0x4547      (region Id)
      //   uint32_t xBytes = stream.Read<uint32_t>();
      //   float x = *reinterpret_cast<float*>(&xBytes);
      //   uint32_t yBytes = stream.Read<uint32_t>();
      //   float y = *reinterpret_cast<float*>(&yBytes);
      //   uint32_t zBytes = stream.Read<uint32_t>();
      //   float z = *reinterpret_cast<float*>(&zBytes);
      //   uint16_t angle = stream.Read<uint16_t>();
      }
    }
  } else if (itemData.haveItemWithId(entity->refObjId)) {
    const auto &item = itemData.getItemById(entity->refObjId);
    const auto itemTypeId = type_id::getTypeId(item);
    if (!categories::kItem.contains(itemTypeId)) {
      throw std::runtime_error("Have an item, but type id does not match");
    }
    entity::Item *itemPtr = dynamic_cast<entity::Item*>(entity.get());
    if (itemPtr == nullptr) {
      throw std::runtime_error("parseSpawn, have an item, but the entity pointer cannot be cast to a Item");
    }
    if (categories::kEquipment.contains(itemTypeId)) {
      uint8_t optLevel = stream.Read<uint8_t>();
    } else if (categories::kExpendable.contains(itemTypeId)) {
      if (categories::kGold.contains(itemTypeId)) {
        uint32_t goldAmount = stream.Read<uint32_t>();
      } else if (categories::kSpecialGoods.contains(itemTypeId) ||
                 categories::kQuestAndEvent.contains(itemTypeId)) {
        uint16_t ownerNameLength = stream.Read<uint16_t>();
        std::string ownerName = stream.Read_Ascii(ownerNameLength);
        LOG(INFO) << "Item is a quest item belonging to " << ownerName;
      }
    }
    itemPtr->globalId = stream.Read<uint32_t>();
    itemPtr->initializePosition(parsePosition(stream));
    sro::Angle angle = stream.Read<sro::Angle>();
    itemPtr->initializeAngle(angle);
    bool hasOwner = stream.Read<uint8_t>();
    if (hasOwner) {
      itemPtr->ownerJId = stream.Read<uint32_t>();
    }
    itemPtr->rarity = stream.Read<sro::entity::ItemRarity>();
  } else if (teleportData.haveTeleportWithId(entity->refObjId)) {
    const auto &teleport = teleportData.getTeleportById(entity->refObjId);
    const auto teleportTypeId = type_id::getTypeId(teleport);
    if (!categories::kStructure.contains(teleportTypeId)) {
      throw std::runtime_error("Have a structure, but type id does not match");
    }
    entity::Portal *portalPtr = dynamic_cast<entity::Portal*>(entity.get());
    if (portalPtr == nullptr) {
      throw std::runtime_error("parseSpawn, have a portal, but the entity pointer cannot be cast to a Portal");
    }
    // PORTALS
    //  STORE
    //  INS_TELEPORTER
    portalPtr->globalId = stream.Read<uint32_t>();
    portalPtr->initializePosition(parsePosition(stream));
    sro::Angle angle = stream.Read<sro::Angle>();
    portalPtr->initializeAngle(angle);

    uint8_t unkByte0 = stream.Read<uint8_t>();
    uint8_t unkByte1 = stream.Read<uint8_t>();
    uint8_t unkByte2 = stream.Read<uint8_t>();
    portalPtr->unkByte3 = stream.Read<uint8_t>();

    if (portalPtr->unkByte3 == 1) {
      // Regular
      uint32_t unkUInt0 = stream.Read<uint32_t>();
      uint32_t unkUInt1 = stream.Read<uint32_t>();
    } else if (portalPtr->unkByte3 == 6) {
      // Dimension Hole
      uint16_t ownerNameLength = stream.Read<uint16_t>();
      std::string ownerName = stream.Read_Ascii(ownerNameLength);
      uint32_t ownerUId = stream.Read<uint32_t>();
    }

    if (unkByte1 == 1) {
      // STORE_EVENTZONE_DEFAULT
      uint32_t unkUint2 = stream.Read<uint32_t>();
      uint8_t unkByte4 = stream.Read<uint8_t>();
    }
  }
  return entity;
}

} // namespace packet::parsing