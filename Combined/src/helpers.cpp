#include "helpers.hpp"
#include "logging.hpp"

#include <silkroad_lib/position.h>
#include <silkroad_lib/position_math.h>

namespace helpers {

float secondsToTravel(const sro::Position &srcPosition, const sro::Position &destPosition, const float currentSpeed) {
  const auto distance = sro::position_math::calculateDistance2d(srcPosition, destPosition);
  return distance / currentSpeed;
}

void initializeInventory(storage::Storage &inventory, uint8_t inventorySize, const std::map<uint8_t, std::shared_ptr<storage::Item>> &inventoryItemMap) {
  // TODO: If there are already items here, there are very few cases when the existing data would be different than the new data
  //  One possible such case is if the server adds an item to our storage, we would need to teleport to see it
  inventory.clear();
  inventory.resize(inventorySize);
  // Guaranteed to have no items
  for (const auto &slotItemPtrPair : inventoryItemMap) {
    inventory.addItem(slotItemPtrPair.first, slotItemPtrPair.second);
  }
}

void printItem(uint8_t slot, const storage::Item *item, const pk2::GameData &gameData) {
  if (item != nullptr) {
    uint16_t quantity = 1;
    const storage::ItemExpendable *itemExpendable = dynamic_cast<const storage::ItemExpendable*>(item);
    if (itemExpendable != nullptr) {
      quantity = itemExpendable->quantity;
    }
    printf("[%3d] %6d (%4d) \"%s\"\n", slot, item->refItemId, quantity, gameData.itemData().getItemById(item->refItemId).codeName128.c_str());
  } else {
    printf("[%3d] --Empty--\n", slot);
  }
}

int toBitNum(packet::enums::AbnormalStateFlag stateFlag) {
  uint32_t num = static_cast<uint32_t>(stateFlag);
  for (uint32_t i=0; i<32; ++i) {
    if (num & (1<<i)) {
      return i;
    }
  }
  throw std::runtime_error("Tried to get bit for a state "+static_cast<int>(stateFlag));
}

packet::enums::AbnormalStateFlag fromBitNum(int n) {
  return static_cast<packet::enums::AbnormalStateFlag>(uint32_t(1) << n);
}

std::shared_ptr<storage::Item> createItemFromScrap(const pk2::ref::ScrapOfPackageItem &itemScrap, const pk2::ref::Item &itemRef) {
  std::shared_ptr<storage::Item> item(storage::newItemByTypeData(itemRef));

  storage::ItemExpendable *itemExpendable;
  storage::ItemEquipment *itemEquipment;
  storage::ItemStone *itemStone;
  storage::ItemCosGrowthSummoner *itemCosGrowthSummoner;
  storage::ItemStorage *itemStorage;
  storage::ItemMonsterCapsule *itemMonsterCapsule;
  storage::ItemCosAbilitySummoner *itemCosAbilitySummoner;
  storage::ItemMagicPop *itemMagicPop;
  if ((itemEquipment = dynamic_cast<storage::ItemEquipment*>(item.get())) != nullptr) {
    itemEquipment->optLevel = itemScrap.optLevel;
    itemEquipment->variance = itemScrap.variance;
    itemEquipment->durability = itemScrap.data;
    for (int i=0; i<itemScrap.magParamNum; ++i) {
      storage::ItemMagicParam param;
      auto paramData = itemScrap.magParams[i];
      uint64_t &data = reinterpret_cast<uint64_t&>(paramData);
      param.type = data & 0xFFFF;
      param.value = (data >> 32) & 0xFFFF;
      itemEquipment->magicParams.emplace_back(std::move(param));
    }
  }
  if ((itemExpendable = dynamic_cast<storage::ItemExpendable*>(item.get())) != nullptr) {
    itemExpendable->quantity = 0; // Arbitrary
  }
  if ((itemStone = dynamic_cast<storage::ItemStone*>(item.get())) != nullptr) {
    // TODO: Verify. No examples in current server
    itemStone->attributeAssimilationProbability = 0;
  }
  if ((itemCosGrowthSummoner = dynamic_cast<storage::ItemCosGrowthSummoner*>(item.get())) != nullptr) {
    // TODO: Verify. Pserver db data is all 0s for Wolf
    itemCosGrowthSummoner->lifeState = storage::CosLifeState::kInactive; // Based on buying then checking data
    itemCosGrowthSummoner->refObjID = 0; // Has no level, no item to ref to
  }
  if ((itemStorage = dynamic_cast<storage::ItemStorage*>(item.get())) != nullptr) {
    // TODO: Verify. No examples in current server
    itemStorage->quantity = 0;
  }
  if ((itemMonsterCapsule = dynamic_cast<storage::ItemMonsterCapsule*>(item.get())) != nullptr) {
    // TODO: Verify. No examples in current server
    itemMonsterCapsule->refObjID = 0; // Total guess
  }
  if ((itemCosAbilitySummoner = dynamic_cast<storage::ItemCosAbilitySummoner*>(item.get())) != nullptr) {
    // TODO: Verify. No examples in current server
    itemCosAbilitySummoner->secondsToRentEndTime = 0; // Based on buying from item mall and looking at data
  }
  if ((itemMagicPop = dynamic_cast<storage::ItemMagicPop*>(item.get())) != nullptr) {
    // TODO: Verify. No examples in current server
  }
  return item;
}

namespace type_id {

namespace {

bool isPill(const pk2::ref::Item &itemInfo) {
  return (itemInfo.typeId1 == 3 && itemInfo.typeId2 == 3 && itemInfo.typeId3 == 2);
}

bool isPotion(const pk2::ref::Item &itemInfo) {
  return (itemInfo.typeId1 == 3 && itemInfo.typeId2 == 3 && itemInfo.typeId3 == 1);
}

} // anonymous namespace

std::tuple<uint8_t,uint8_t,uint8_t,uint8_t> splitTypeId(const uint16_t typeId) {
  const uint8_t typeId1 = (typeId >> 2) & 0b111;
  const uint8_t typeId2 = (typeId >> 5) & 0b11;
  const uint8_t typeId3 = (typeId >> 7) & 0b1111;
  const uint8_t typeId4 = (typeId >> 11);
  return {typeId1, typeId2, typeId3, typeId4};
}

} // namespace type_id

} // namespace helpers