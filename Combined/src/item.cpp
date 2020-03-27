#include "item.hpp"

#include <iostream>
#include <iomanip>

namespace item {

uint16_t Item::typeData() const {
  uint16_t itemTypeData = 0;
  if (itemInfo != nullptr) {
    itemTypeData |= itemInfo->cashItem;
    itemTypeData |= (itemInfo->bionic << 1);
    itemTypeData |= (itemInfo->typeId1 << 2);
    itemTypeData |= (itemInfo->typeId2 << 5);
    itemTypeData |= (itemInfo->typeId3 << 7);
    itemTypeData |= (itemInfo->typeId4 << 11);
  }
  return itemTypeData;
}

Item::Item(ItemType t) : type(t) {}
Item::~Item() {}
ItemEquipment::ItemEquipment() : Item(ItemType::kItemEquipment) {}
ItemCosGrowthSummoner::ItemCosGrowthSummoner() : Item(ItemType::kItemCosGrowthSummoner) {}
ItemCosGrowthSummoner::ItemCosGrowthSummoner(ItemType type) : Item(type) {}
ItemCosAbilitySummoner::ItemCosAbilitySummoner() : ItemCosGrowthSummoner(ItemType::kItemCosAbilitySummoner) {}
ItemMonsterCapsule::ItemMonsterCapsule() : Item(ItemType::kItemMonsterCapsule) {}
ItemStorage::ItemStorage() : Item(ItemType::kItemStorage) {}
ItemExpendable::ItemExpendable() : Item(ItemType::kItemExpendable) {}
ItemExpendable::ItemExpendable(ItemType type) : Item(type) {}
ItemStone::ItemStone() : ItemExpendable(ItemType::kItemStone) {}
ItemMagicPop::ItemMagicPop() : ItemExpendable(ItemType::kItemMagicPop) {}

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
  std::cout << "stackCount: " << item.stackCount << '\n';
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

} // namespace item