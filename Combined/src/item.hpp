#ifndef ITEM_HPP__
#define ITEM_HPP__

#include "../../common/itemInfo.hpp"

#include <cstdint>
#include <vector>

namespace item {

class Item {
public:
  uint32_t refItemId;
  const pk2::media::Item *itemInfo{nullptr};
  virtual ~Item() = 0; // Make base type polymorphic and uninstantiable
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
  uint8_t optLevel;
  uint64_t variance;
  uint32_t durability; // "Data" (also can be devil spirit's (nasrun) secondsToRentEndTime)
  std::vector<ItemMagicParam> magicParams;
  std::vector<SocketOptionData> socketOptions;
  std::vector<AdvancedElixirOptionData> advancedElixirOptions;
};

// CGItemCOSSummoner, ITEM_COS_P
class ItemCosGrowthSummoner : public Item {
public:
  CosLifeState lifeState;
  uint32_t refObjID;
  std::string name;
  std::vector<CosJob> jobs;  
};

// CGItemCOSSummoner, ITEM_COS_P
class ItemCosAbilitySummoner : public ItemCosGrowthSummoner {
public:
  uint32_t secondsToRentEndTime;
};

// CGItemMonsterCapsule, ITEM_ETC_TRANS_MONSTER (rogue mask)
class ItemMonsterCapsule : public Item {
public:
  uint32_t refObjID;
};

// CGItemStorage, MAGIC_CUBE
class ItemStorage : public Item {
public:
  uint32_t quantity; // Do not confuse with StackCount, this indicates the amount of elixirs in the cube
};

// CGItemExpendable, ITEM_ETC
class ItemExpendable : public Item {
public:
  uint16_t stackCount;
};

// MAGICSTONE, ATTRSTONE
class ItemStone : public ItemExpendable {
public:
  uint8_t attributeAssimilationProbability; // stored in _Items.OptLevel on the server side
};

// ITEM_MALL_GACHA_CARD_WIN, ITEM_MALL_GACHA_CARD_LOSE
class ItemMagicPop : public ItemExpendable {
public:
  std::vector<ItemMagicParam> magicParams;
};

void print(const Item *item);

} // namespace item

#endif // ITEM_HPP__