#include "serverAgentCosData.hpp"

#include "packet/parsing/commonParsing.hpp"

namespace packet::parsing {

ServerAgentCosData::ServerAgentCosData(const PacketContainer &packet, const sro::pk2::CharacterData &characterData, const sro::pk2::ItemData &itemData) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  globalId_ = stream.Read<uint32_t>();
  uint32_t refObjId = stream.Read<uint32_t>();
  uint32_t currHp = stream.Read<uint32_t>();
  uint32_t maxHp = stream.Read<uint32_t>();

  const auto &cosRef = characterData.getCharacterById(refObjId);
  if (!(cosRef.typeId1 == 1 && cosRef.typeId2 == 2 && cosRef.typeId3 == 3)) {
    throw std::runtime_error("Cos data given refObjId which does not have matching type data for a Cos");
  }
  typeId4_ = cosRef.typeId4;

  if (isAbilityPet()) {
    // Ability/pickpet
    uint32_t settings = stream.Read<uint32_t>();
    uint16_t nameLength = stream.Read<uint16_t>();
    std::string name = stream.Read_Ascii(nameLength);
    inventorySize_ = stream.Read<uint8_t>();
    uint8_t inventoryItemCount = stream.Read<uint8_t>();
    for (int i=0; i<inventoryItemCount; ++i) {
      uint8_t slotNum = stream.Read<uint8_t>();
      inventoryItemMap_.insert(std::pair<uint8_t, std::shared_ptr<storage::Item>>(slotNum, parseGenericItem(stream, itemData)));
    }
    ownerGlobalId_ = stream.Read<uint32_t>();
    uint8_t slotInOwnerInventory = stream.Read<uint8_t>();
  }

  // Get type ID from refObjId
  // if (ability) {
  //   uint32_t settings = stream.Read<uint32_t>();
  //   // ...
  // } else {
  //   // ...
  // }
  // 4	uint	COS.UniqueID
  // 4	uint	COS.RefObjID
  // 4	uint	COS.CurHP
  // 4	uint	COS.MaxHP
  // 4	uint	COS.Settings
  // 2	ushort	COS.Name.Length
  // *	string	COS.Name
  // 1	byte	COS.Inventory.Size
  // 1	byte	COS.Inventory.ItemCount
  // foreach(item)
  // {
  //   1	byte	item.Slot
  //   4	uint	item.RentType
  //   *	object	item.<genericRentInfo>
  //   4	uint	item.RefObjID
  //   *	object	item.<genericItemData>	
  // }
  // 4	uint	Owner.UniqueID
  // 1	byte	SourceSlot	//slot of summon scroll in owners inventory.
  // updatePointsType_ = static_cast<packet::enums::UpdatePointsType>(stream.Read<uint8_t>());
  // if (updatePointsType_ == packet::enums::UpdatePointsType::kGold) {
  //   gold_ = stream.Read<uint64_t>();
  //   isDisplayed_ = stream.Read<uint8_t>();
  // } else if (updatePointsType_ == packet::enums::UpdatePointsType::kSp) {
  //   skillPoints_ = stream.Read<uint32_t>();
  //   isDisplayed_ = stream.Read<uint8_t>();
  // } else if (updatePointsType_ == packet::enums::UpdatePointsType::kStatPoint) {
  //   uint16_t StatPoints = stream.Read<uint16_t>();
  // } else if (updatePointsType_ == packet::enums::UpdatePointsType::kHwan) {
  //   uint8_t HwanCount = stream.Read<uint8_t>();
  //   uint32_t Source_UniqueID = stream.Read<uint32_t>();
  // } else if (updatePointsType_ == packet::enums::UpdatePointsType::kAp) {
  //   uint32_t APPoint = stream.Read<uint32_t>();
  // }
}

uint32_t ServerAgentCosData::globalId() const {
  return globalId_;
}

bool ServerAgentCosData::isAbilityPet() const {
  return (typeId4_ == 4);
}

uint8_t ServerAgentCosData::inventorySize() const {
  return inventorySize_;
}

const std::map<uint8_t, std::shared_ptr<storage::Item>>& ServerAgentCosData::inventoryItemMap() const {
  return inventoryItemMap_;
}

uint32_t ServerAgentCosData::ownerGlobalId() const {
  return ownerGlobalId_;
}

} // namespace packet::parsing