#include "item.hpp"

#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

namespace storage {

type_id::TypeId Item::typeId() const {
  if (itemInfo == nullptr) {
    throw std::runtime_error("Dont have a pointer to our item data");
  }
  return (itemInfo->cashItem & 0b1) |
         ((itemInfo->bionic & 0b1) << 1) |
         ((itemInfo->typeId1 & 0b111) << 2) |
         ((itemInfo->typeId2 & 0b11) << 5) |
         (static_cast<type_id::TypeId>(itemInfo->typeId3 & 0b1111) << 7) |
         (static_cast<type_id::TypeId>(itemInfo->typeId4 & 0b11111) << 11);
}

bool Item::isA(const type_id::TypeCategory &typeCategory) const {
  return typeCategory.contains(typeId());
}

bool Item::isOneOf(const std::vector<type_id::TypeCategory> &typeCategories) const {
  const auto thisItemTypeData = typeId();
  for (const auto &category : typeCategories) {
    if (category.contains(thisItemTypeData)) {
      return true;
    }
  }
  return false;
}

Item::Item(ItemType t) : type(t) {}
Item::~Item() {}

ItemEquipment::ItemEquipment() : Item(ItemType::kItemEquipment) {}

bool ItemEquipment::repairInvalid(const pk2::GameData &gameData) const {
  for (const auto &blue : magicParams) {
    const auto &magicOption = gameData.magicOptionData().getMagicOptionById(blue.type);
    // Rather than string comparisons, we could use the 2-4 character string from Param1, but string comparison is actually what the client uses.
    if (magicOption.mOptName128 == "MATTR_NOT_REPARABLE") {
      return true;
    }
  }
  return false;
}

uint32_t ItemEquipment::maxDurability(const pk2::GameData &gameData) const {
  // The durability is the first 5 bits within the variance
  // TODO: Extracting this from the variance should be done in a more general way
  const uint64_t kDurabilityMask{0b11111};
  const auto durabilityRange = itemInfo->dur_U - itemInfo->dur_L;
  const auto currentDurabilityWhiteVarianceValue = variance & kDurabilityMask;
  // Intentionally truncate and cast to an int
  uint32_t currentDurabilityWhiteValue = itemInfo->dur_L + (durabilityRange * currentDurabilityWhiteVarianceValue) / static_cast<double>(kDurabilityMask);

  // Does this item have a blue/red durability magic option?
  int percentDurabilityPercentIncrease = 0;
  for (const auto &blue : magicParams) {
    const auto &magicOption = gameData.magicOptionData().getMagicOptionById(blue.type);
    // Rather than string comparisons, we could use the 2-4 character string from Param1, but string comparison is actually what the client uses.
    if (magicOption.mOptName128 == "MATTR_DUR") {
      percentDurabilityPercentIncrease += blue.value;
    } else if (magicOption.mOptName128 == "MATTR_DEC_MAXDUR") {
      percentDurabilityPercentIncrease -= blue.value;
    } else if (magicOption.mOptName128 == "MATTR_NOT_REPARABLE") {
      percentDurabilityPercentIncrease += magicOption.param3;
    }
  }
  if (percentDurabilityPercentIncrease != 0) {
    currentDurabilityWhiteValue = currentDurabilityWhiteValue * (100 + percentDurabilityPercentIncrease) / 100.0;
  }

  // TODO: This calculation isn't perfect. Sometimes it's off by 1. Might be a rounding error?
  return currentDurabilityWhiteValue;
}

int ItemEquipment::degree() const {
  return (itemInfo->itemClass-1)/3 + 1;
}

ItemCosGrowthSummoner::ItemCosGrowthSummoner() : Item(ItemType::kItemCosGrowthSummoner) {}
ItemCosGrowthSummoner::ItemCosGrowthSummoner(ItemType type) : Item(type) {}
ItemCosAbilitySummoner::ItemCosAbilitySummoner() : ItemCosGrowthSummoner(ItemType::kItemCosAbilitySummoner) {}
ItemMonsterCapsule::ItemMonsterCapsule() : Item(ItemType::kItemMonsterCapsule) {}
ItemStorage::ItemStorage() : Item(ItemType::kItemStorage) {}
ItemExpendable::ItemExpendable() : Item(ItemType::kItemExpendable) {}
ItemExpendable::ItemExpendable(ItemType type) : Item(type) {}
ItemStone::ItemStone() : ItemExpendable(ItemType::kItemStone) {}
ItemMagicPop::ItemMagicPop() : ItemExpendable(ItemType::kItemMagicPop) {}

std::shared_ptr<storage::Item> newItemByTypeData(const pk2::ref::Item &item) {
  std::shared_ptr<storage::Item> storageItemPtr;
  if (item.typeId1 == 3) {
    if (item.typeId2 == 1) {
      // CGItemEquip
      storageItemPtr.reset(new ItemEquipment());
    } else if (item.typeId2 == 2) {
      if (item.typeId3 == 1) {                                
        // CGItemCOSSummoner
        if (item.typeId4 == 2) {
          storageItemPtr.reset(new ItemCosAbilitySummoner());
        } else {
          storageItemPtr.reset(new ItemCosGrowthSummoner());
        }
      } else if (item.typeId3 == 2) {
        // CGItemMonsterCapsule (rogue mask)
        storageItemPtr.reset(new ItemMonsterCapsule());
      } else if (item.typeId3 == 3) {
        // CGItemStorage
        storageItemPtr.reset(new ItemStorage());
      }
    } else if (item.typeId2 == 3) {
      bool itemCreated = false;
      // CGItemExpendable
      if (item.typeId3 == 11) {
        if (item.typeId4 == 1 || item.typeId4 == 2) {
          // MAGICSTONE, ATTRSTONE
          storageItemPtr.reset(new ItemStone());
          itemCreated = true;
        }
      } else if (item.typeId3 == 14 && item.typeId4 == 2) {
        // Magic pop
        storageItemPtr.reset(new ItemMagicPop());
        itemCreated = true;
      }
      if (!itemCreated) {
        // Other expendable
        storageItemPtr.reset(new ItemExpendable());
      }
    }
  }

  if (storageItemPtr) {
    // Set base info
    storageItemPtr->refItemId = item.id;
    storageItemPtr->itemInfo = &item;
  }

  return storageItemPtr;
}

std::shared_ptr<Item> cloneItem(const Item *item) {
  switch (item->type) {
    case ItemType::kItemEquipment:
      return std::shared_ptr<Item>(new ItemEquipment(*dynamic_cast<const ItemEquipment*>(item)));
    case ItemType::kItemCosGrowthSummoner:
      return std::shared_ptr<Item>(new ItemCosGrowthSummoner(*dynamic_cast<const ItemCosGrowthSummoner*>(item)));
    case ItemType::kItemCosAbilitySummoner:
      return std::shared_ptr<Item>(new ItemCosAbilitySummoner(*dynamic_cast<const ItemCosAbilitySummoner*>(item)));
    case ItemType::kItemMonsterCapsule:
      return std::shared_ptr<Item>(new ItemMonsterCapsule(*dynamic_cast<const ItemMonsterCapsule*>(item)));
    case ItemType::kItemStorage:
      return std::shared_ptr<Item>(new ItemStorage(*dynamic_cast<const ItemStorage*>(item)));
    case ItemType::kItemExpendable:
      return std::shared_ptr<Item>(new ItemExpendable(*dynamic_cast<const ItemExpendable*>(item)));
    case ItemType::kItemStone:
      return std::shared_ptr<Item>(new ItemStone(*dynamic_cast<const ItemStone*>(item)));
    case ItemType::kItemMagicPop:
      return std::shared_ptr<Item>(new ItemMagicPop(*dynamic_cast<const ItemMagicPop*>(item)));
  }
  return {};
}

namespace {

void print(const ItemEquipment &item) {
  LOG(INFO) << "refItemId: " << item.refItemId;
  LOG(INFO) << "optLevel: " << static_cast<int>(item.optLevel);
  LOG(INFO) << "variance: " << item.variance;
  LOG(INFO) << "durability: " << item.durability;
  LOG(INFO) << absl::StreamFormat("magicParams: [%s]", absl::StrJoin(item.magicParams, ", ", [](std::string *out, const auto magicParam) {
    out->append(absl::StrFormat("(type: %d,value: %d)", magicParam.type, magicParam.value));
  }));
  LOG(INFO) << absl::StreamFormat("socketOptions: [%s]", absl::StrJoin(item.socketOptions, ", ", [](std::string *out, const auto socketOption) {
    out->append(absl::StrFormat("(slot: %d, id: %d, nParam1: %d)", socketOption.slot, socketOption.id, socketOption.nParam1));
  }));
  LOG(INFO) << absl::StreamFormat("advancedElixirOptions: [%s]", absl::StrJoin(item.advancedElixirOptions, ", ", [](std::string *out, const auto advancedElixirOption) {
    out->append(absl::StrFormat("(slot: %d, id: %d, optValue: %d)", advancedElixirOption.slot, advancedElixirOption.id, advancedElixirOption.optValue));
  }));
}

void print(const ItemCosGrowthSummoner *item) {
  LOG(INFO) << "refItemId: " << item->refItemId;
  LOG(INFO) << "lifeState: " << static_cast<int>(item->lifeState);
  LOG(INFO) << "refObjID: " << item->refObjID;
  LOG(INFO) << "name: " << std::quoted(item->name);
  const ItemCosAbilitySummoner *ptr;
  if (ptr = dynamic_cast<const ItemCosAbilitySummoner*>(item)) {
    LOG(INFO) << "secondsToRentEndTime: " << ptr->secondsToRentEndTime;
  }
  LOG(INFO) << absl::StreamFormat("jobs: [%s]", absl::StrJoin(item->jobs, ", ", [](std::string *out, const auto job) {
    out->append(absl::StrFormat("(category: %d, jobId: %d, timeToKeep: %d, data1: %d, data2: %d)", job.category, job.jobId, job.timeToKeep, job.data1, job.data2));
  }));
}

void print(const ItemCosGrowthSummoner &item) {
  print(&item);
}

void print(const ItemCosAbilitySummoner &item) {
  print(&item);
}

void print(const ItemMonsterCapsule &item) {
  LOG(INFO) << "refItemId: " << item.refItemId;
  LOG(INFO) << "refObjID: " << item.refObjID;
}

void print(const ItemStorage &item) {
  LOG(INFO) << "refItemId: " << item.refItemId;
  LOG(INFO) << "quantity: " << item.quantity;
}

void print(const ItemExpendable &item) {
  LOG(INFO) << "refItemId: " << item.refItemId;
  LOG(INFO) << "stackCount: " << item.quantity;
}

void print(const ItemStone &item) {
  LOG(INFO) << "refItemId: " << item.refItemId;
  LOG(INFO) << "attributeAssimilationProbability: " << static_cast<int>(item.attributeAssimilationProbability);
}

void print(const ItemMagicPop &item) {
  LOG(INFO) << "refItemId: " << item.refItemId;
  LOG(INFO) << absl::StreamFormat("magicParams: [%s]", absl::StrJoin(item.magicParams, ", ", [](std::string *out, const auto magicParam){
    out->append(absl::StrFormat("(type: %d,value: %d)", magicParam.type, magicParam.value));
  }));
}

} // anonymous namespace

void print(const Item *item) {
  const ItemEquipment *equipment;
  const ItemCosAbilitySummoner *cosAbilitySummoner;
  const ItemCosGrowthSummoner *cosGrowthSummoner;
  const ItemMonsterCapsule *monsterCapsule;
  const ItemStorage *storage;
  const ItemStone *stone;
  const ItemMagicPop *magicPop;
  const ItemExpendable *expendable;
  if ((equipment = dynamic_cast<const ItemEquipment*>(item)) != nullptr) {
    print(*equipment);
  } else if ((cosAbilitySummoner = dynamic_cast<const ItemCosAbilitySummoner*>(item)) != nullptr) {
    print(*cosAbilitySummoner);
  } else if ((cosGrowthSummoner = dynamic_cast<const ItemCosGrowthSummoner*>(item)) != nullptr) {
    print(*cosGrowthSummoner);
  } else if ((monsterCapsule = dynamic_cast<const ItemMonsterCapsule*>(item)) != nullptr) {
    print(*monsterCapsule);
  } else if ((storage = dynamic_cast<const ItemStorage*>(item)) != nullptr) {
    print(*storage);
  } else if ((stone = dynamic_cast<const ItemStone*>(item)) != nullptr) {
    print(*stone);
  } else if ((magicPop = dynamic_cast<const ItemMagicPop*>(item)) != nullptr) {
    print(*magicPop);
  } else if ((expendable = dynamic_cast<const ItemExpendable*>(item)) != nullptr) {
    print(*expendable);
  }
}

} // namespace storage