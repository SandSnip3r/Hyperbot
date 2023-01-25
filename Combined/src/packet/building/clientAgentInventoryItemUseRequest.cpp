#include "clientAgentInventoryItemUseRequest.hpp"

namespace packet::building {

PacketContainer ClientAgentInventoryItemUseRequest::packet(sro::scalar_types::StorageIndexType inventoryIndex, type_id::TypeId typeId) {
  StreamUtility stream;
  stream.Write(inventoryIndex);
  stream.Write(typeId);
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
  /*
  // TODO:
  //TID1 | TID2 | TID3 | TID4 | Name
  //3    | 3    | 1    | 4    | ITEM_ETC_POTION_COS_HP
  //3    | 3    | 1    | 9    | ITEM_ETC_POTION_COS_HGP
  //3    | 3    | 1    | 10   | ITEM_FORT_REPAIR_KIT
  //3    | 3    | 2    | 7    | ITEM_ETC_CURE_COS_ALL
  //3    | 3    | 6    | 2    | ITEM_FORT_SHOCK_BOMB
  clientUseItemData.Write<uint32_t>(targetGId);

  //3    | 3    | 1    | 6    | ITEM_ETC_POTION_COS_REVIVE
  //3    | 3    | 13   | 8    | ITEM_ETC_SPECIAL_EQUIP_TRANSGENDER
  //3    | 3    | 13   | 11   | ITEM_ETC_SPECIAL_EQUIP_REINFORCE
  //3    | 3    | 13   | 12   | ITEM_COS_P_EXTENSION
  //3    | 3    | 13   | 15   | ITEM_PET_HELPER
  //3    | 3    | 13   | 16   | ITEM_ETC_NASRUN_EXTENSION
  clientUseItemData.Write<uint8_t>(targetSlot);

  //3    | 3    | 3    | 3    | ITEM_ETC_SCROLL_REVERSE_RETURN
  clientUseItemData.Write<uint8_t>(reverseOption); // 2 = last recall point, 3 = last death location, 7 = location on map
  if (reverseOption == 7) {
    clientUseItemData.Write<uint32_t>(optionalTeleportId);
  }

  //3    | 3    | 3    | 5    | ITEM_ETC_SCROLL_CHATTING
  clientUseItemData.Write<uint16_t>(message.size());
  clientUseItemData.Write_Ascii(message);

  //3    | 3    | 3    | 6    | ITEM_FORT_TABLET
  //3    | 3    | 3    | 7    | ITEM_ETC_SCROLL_COS_GUARD
  //3    | 3    | 3    | 9    | ITEM_FORT_FLAG
  clientUseItemData.Write<uint32_t>(fortressID);
  clientUseItemData.Write<uint16_t>(rid);
  clientUseItemData.Write<uint32_t>(xOffset);
  clientUseItemData.Write<uint32_t>(yOffset);
  clientUseItemData.Write<uint32_t>(zOffset);

  //3    | 3    | 3    | 8    | ITEM_FORT_MANUAL
  clientUseItemData.Write<uint32_t>(fortressID);
  clientUseItemData.Write<uint32_t>(refEventID);

  //3    | 3    | 12   | 2    | ITEM_ETC_GUILD_CREST
  //NOTE: ONLY ON ITEM_ETC_GUILD_CREST_USER & ITEM_ETC_UNION_CREST_USER
  256 byte[]  imagePayload

  //3    | 3    | 13   | 9    | ITEM_ETC_SKIN_CHANGE
  clientUseItemData.Write<uint32_t>(refObjId);
  clientUseItemData.Write<uint8_t>(scale);
  */
}

} // namespace packet::building