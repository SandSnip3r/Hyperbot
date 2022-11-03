#include "commonBuilding.hpp"

namespace packet::building {

void writeGenericItem(StreamUtility &stream, const storage::Item &item) {
  if (!item.itemInfo) {
    throw std::runtime_error("Trying to build item, but item info is null");
  }
  // 01 // success
  // 0e // add item by server
  // 1e // target slot
  // 00 // reason
  // 00 00 00 00 // rent type
  // 00 00 00 08 // ref id (seems wrong?)
  // 00 00 // stack count
  // 00 00 00 // ????

  // [39972761] StreamUtility::Write:109: Writing something of size 1
  // [39972762] StreamUtility::Write:109: Writing something of size 1
  // [39972763] StreamUtility::Write:109: Writing something of size 1
  // [39972763] StreamUtility::Write:109: Writing something of size 4
  // [39972764] StreamUtility::Write:109: Writing something of size 4
  // [39972765] StreamUtility::Write:109: Writing something of size 4
  // [39972766] StreamUtility::Write:109: Writing something of size 2
  // 01 0e 1e 00 00 00 00 00 00 00 00 08 00 00 00 32 00

  stream.Write<uint32_t>(0); // RentType
  stream.Write<>(item.refItemId);

  if(item.itemInfo->typeId1 == 3) {
    //CGItem        
    if(item.itemInfo->typeId2 == 1) {
      // TODO: !!!!!!
      throw std::runtime_error("Not yet implemented, would crash");
      // //CGItemEquip
      // 1   byte    item.OptLevel
      // 8   ulong   item.Variance
      // 4   uint    item.Data       //Durability
      // 1   byte    item.MagParamNum
      // for(int paramIndex = 0; paramIndex < item.MagParamNum; paramIndex++) {
      //   4   uint    magParam.Type
      //   4   uint    magParam.Value                
      // }

      // 1   byte    bindingOptionType   //1 = Socket
      // 1   byte    bindingOptionCount
      // for (int bindingOptionIndex = 0; bindingOptionIndex < bindingOptionCount; bindingOptionIndex++) {
      //   1   byte bindingOption.Slot
      //   4   uint bindingOption.ID
      //   4   uint bindingOption.nParam1
      // }

      // 1   byte    bindingOptionType   //2 = Advanced elixir
      // 1   byte    bindingOptionCount
      // for (int bindingOptionIndex = 0; bindingOptionIndex < bindingOptionCount; bindingOptionIndex++) {
      //   1   byte bindingOption.Slot
      //   4   uint bindingOption.ID
      //   4   uint bindingOption.OptValue
      // }            
    } else if(item.itemInfo->typeId2 == 2) {
      // TODO: !!!!!!!!
      throw std::runtime_error("Not yet implemented, would crash");
      // //CGItemContainer        
      // if(item.itemInfo->typeId3 == 1) {                                
      //   //CGItemCOSSummoner
      //   1   byte    State   // 1 = Inactive, 2 = Summoned, 3 = Active, 4 = Dead
      //   if(State != 1) {                    
      //     4   uint    RefObjID
      //     2   ushort  Name.Length
      //     *   string  Name
      //     if(item.itemInfo->typeId4 == 4) {
      //       //ITEM_COS_P (Ability)
      //       4   uint    SecondsToRentEndTime
      //     }
      //     1   byte    timedJobCount
      //     foreach(timedJobCount) {
      //       //_TimedJobForPet
      //       1   byte    Category
      //       4   uint    JobID   //Category3 = RefSkillID, Category5 = RefItemID
      //       4   uint    TimeToKeep
      //       if(Category == 5) {
      //         4   uint    Data1
      //         1   byte    Data2
      //       }
      //     }
      //   }
      // } else if(item.itemInfo->typeId3 == 2) {
      //   //CGItemMonsterCapsule
      //   4   uint    RefObjID
      // } else if(item.itemInfo->typeId3 == 3) {
      //   //CGItemStorage
      //   4   uint    Quantity        //Do not confuse with StackCount, this indicates the amount of elixirs in the cube
      // }
    } else if(item.itemInfo->typeId2 == 3) {
      //CGItemExpendable
      const auto *itemAsExpendable = dynamic_cast<const storage::ItemExpendable*>(&item);
      if (itemAsExpendable == nullptr) {
        throw std::runtime_error("We thought we had an expendable item, but we do not");
      }

      stream.Write<>(itemAsExpendable->quantity); // Stack count

      if(item.itemInfo->typeId3 == 11) {
        if(item.itemInfo->typeId4 == 1 || item.itemInfo->typeId4 == 2) {
          // TODO: !!!!!!
          throw std::runtime_error("Not yet implemented, would crash");
          // //MAGICSTONE, ATTRSTONE
          // 1   byte    AttributeAssimilationProbability //stored in OptLevel
        }
      } else if(item.itemInfo->typeId3 == 14 && item.itemInfo->typeId4 == 2) {
        // TODO: !!!!!!
        throw std::runtime_error("Not yet implemented, would crash");
        // //ITEM_MALL_GACHA_CARD_WIN
        // //ITEM_MALL_GACHA_CARD_LOSE
        // 1   byte    item.MagParamCount
        // for (int paramIndex = 0; paramIndex < MagParamNum; paramIndex++) {
        //   4   uint magParam.Type
        //   4   uint magParam.Value
        // }
      }
    }
  }
}

void writePosition(StreamUtility &stream, const sro::Position &position) {
  stream.Write(position.regionId());
  stream.Write(position.xOffset());
  stream.Write(position.yOffset());
  stream.Write(position.zOffset());
}

} // namespace packet::building