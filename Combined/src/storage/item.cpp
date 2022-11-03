#include "item.hpp"
#include "logging.hpp"

#include <iostream>
#include <iomanip>

namespace storage {

type_id::TypeId Item::typeData() const {
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
  return typeCategory.contains(typeData());
}

bool Item::isOneOf(const std::vector<type_id::TypeCategory> &typeCategories) const {
  const auto thisItemTypeData = typeData();
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
  std::cout << "refItemId: " << item.refItemId << '\n';
  std::cout << "optLevel: " << static_cast<int>(item.optLevel) << '\n';
  std::cout << "variance: " << item.variance << '\n';
  std::cout << "durability: " << item.durability << '\n';
  std::cout << "magicParams: [";
  for (auto &i : item.magicParams) {
    std::cout << "(type: " << i.type << ',';
    std::cout << "value: " << i.value << "), ";
  }
  std::cout << "]\n";
  std::cout << "socketOptions: [";
  for (auto &i : item.socketOptions) {
    std::cout << "(slot: " << static_cast<int>(i.slot) << ',';
    std::cout << "id: " << i.id << ',';
    std::cout << "nParam1: " << i.nParam1 << "), ";
  }
  std::cout << "]\n";
  std::cout << "advancedElixirOptions: [";
  for (auto &i : item.advancedElixirOptions) {
    std::cout << "(slot: " << static_cast<int>(i.slot) << ',';
    std::cout << "id: " << i.id << ',';
    std::cout << "optValue: " << i.optValue << "), ";
  }
  std::cout << "]\n";
}

void print(const ItemCosGrowthSummoner *item) {
  std::cout << "refItemId: " << item->refItemId << '\n';
  std::cout << "lifeState: " << static_cast<int>(item->lifeState) << '\n';
  std::cout << "refObjID: " << item->refObjID << '\n';
  std::cout << "name: " << std::quoted(item->name) << '\n';
  const ItemCosAbilitySummoner *ptr;
  if (ptr = dynamic_cast<const ItemCosAbilitySummoner*>(item)) {
    std::cout << "secondsToRentEndTime: " << ptr->secondsToRentEndTime << '\n';
  }
  std::cout << "jobs: [";
  for (auto &i : item->jobs) {
    std::cout << "(category: " << static_cast<int>(i.category) << ',';
    std::cout << "jobId: " << i.jobId << ',';
    std::cout << "timeToKeep: " << i.timeToKeep << ',';
    std::cout << "data1: " << i.data1 << ',';
    std::cout << "data2: " << static_cast<int>(i.data2) << "), ";
  }
  std::cout << "]\n";
}

void print(const ItemCosGrowthSummoner &item) {
  print(&item);
}

void print(const ItemCosAbilitySummoner &item) {
  print(&item);
}

void print(const ItemMonsterCapsule &item) {
  std::cout << "refItemId: " << item.refItemId << '\n';
  std::cout << "refObjID: " << item.refObjID << '\n';
}

void print(const ItemStorage &item) {
  std::cout << "refItemId: " << item.refItemId << '\n';
  std::cout << "quantity: " << item.quantity << '\n';
}

void print(const ItemExpendable &item) {
  std::cout << "refItemId: " << item.refItemId << '\n';
  std::cout << "stackCount: " << item.quantity << '\n';
}

void print(const ItemStone &item) {
  std::cout << "refItemId: " << item.refItemId << '\n';
  std::cout << "attributeAssimilationProbability: " << static_cast<int>(item.attributeAssimilationProbability) << '\n';
}

void print(const ItemMagicPop &item) {
  std::cout << "refItemId: " << item.refItemId << '\n';
  std::cout << "magicParams: [";  
  for (auto &i : item.magicParams) {
    std::cout << "(type: " << i.type << ',';
    std::cout << "value: " << i.value << "),";
  }
  std::cout << "]\n";
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