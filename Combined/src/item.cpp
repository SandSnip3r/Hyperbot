#include "item.hpp"

#include <iostream>
#include <iomanip>

namespace item {

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

} // namespace item