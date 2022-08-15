#ifndef ITEM_HPP__
#define ITEM_HPP__

#include "pk2/gameData.hpp"
#include "../../../common/pk2/ref/item.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace storage {

enum class ItemType {
  kItemEquipment,
  kItemCosGrowthSummoner,
  kItemCosAbilitySummoner,
  kItemMonsterCapsule,
  kItemStorage,
  kItemExpendable,
  kItemStone,
  kItemMagicPop
};

class Item {
public:
  uint32_t refItemId;
  const ItemType type;
  const pk2::ref::Item *itemInfo{nullptr};
  uint16_t typeData() const;
  virtual ~Item() = 0; // Make base type polymorphic and uninstantiable
protected:
  Item(ItemType t);
};

struct ItemMagicParam {
public:
  uint32_t type, value;
};

struct SocketOptionData {
public:
  uint8_t slot;
  uint32_t id, nParam1;
};

struct AdvancedElixirOptionData {
public:
  uint8_t slot;
  uint32_t id, optValue;
};

enum class CosLifeState : uint8_t {
  kInactive = 1,
  kSummoned = 2,
  kActive = 3,
  kDead = 4
};

struct CosJob {
public:
  uint8_t category;
  uint32_t jobId; // category3 = RefSkillId, category5 = RefItemId
  uint32_t timeToKeep;
  uint32_t data1;
  uint8_t data2;
};

// CGItemEquip, ITEM_CH, ITEM_EU, AVATAR_
class ItemEquipment : public Item {
public:
  ItemEquipment();
  uint8_t optLevel;
  uint64_t variance;
  uint32_t durability; // "Data" (also can be devil spirit's (nasrun) secondsToRentEndTime)
  std::vector<ItemMagicParam> magicParams;
  std::vector<SocketOptionData> socketOptions;
  std::vector<AdvancedElixirOptionData> advancedElixirOptions;
  bool repairInvalid(const pk2::GameData &gameData) const;
  uint32_t maxDurability(const pk2::GameData &gameData) const;
};

// CGItemCOSSummoner, ITEM_COS_P
class ItemCosGrowthSummoner : public Item {
public:
  ItemCosGrowthSummoner();
  CosLifeState lifeState;
  uint32_t refObjID; // Wolfs, for example, have an object for each level wolf, so you can know the stats of the wolf
  std::string name;
  std::vector<CosJob> jobs;  
protected:
  ItemCosGrowthSummoner(ItemType type);
};

// CGItemCOSSummoner, ITEM_COS_P
class ItemCosAbilitySummoner : public ItemCosGrowthSummoner {
public:
  ItemCosAbilitySummoner();
  uint32_t secondsToRentEndTime;
};

// CGItemMonsterCapsule, ITEM_ETC_TRANS_MONSTER (rogue mask)
class ItemMonsterCapsule : public Item {
public:
  ItemMonsterCapsule();
  uint32_t refObjID; // The monster which this mask transforms into
};

// CGItemStorage, MAGIC_CUBE
class ItemStorage : public Item {
public:
  ItemStorage();
  uint32_t quantity; // This indicates the amount of elixirs in the cube
};

// CGItemExpendable, ITEM_ETC
class ItemExpendable : public Item {
public:
  ItemExpendable();
  uint16_t quantity;
protected:
  ItemExpendable(ItemType type);
};

// MAGICSTONE, ATTRSTONE
class ItemStone : public ItemExpendable {
public:
  ItemStone();
  uint8_t attributeAssimilationProbability; // stored in _Items.OptLevel on the server side
};

// ITEM_MALL_GACHA_CARD_WIN, ITEM_MALL_GACHA_CARD_LOSE
class ItemMagicPop : public ItemExpendable {
public:
  ItemMagicPop();
  std::vector<ItemMagicParam> magicParams;
};

std::shared_ptr<storage::Item> newItemByTypeData(const pk2::ref::Item &item);
std::shared_ptr<storage::Item> cloneItem(const storage::Item *item);

void print(const Item *item);

} // namespace storage

#endif // ITEM_HPP__