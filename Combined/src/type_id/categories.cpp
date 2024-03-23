#include "categories.hpp"

namespace type_id::categories {

const TypeCategory kCharacter{static_cast<uint8_t>(1)};
  const TypeCategory kPlayerCharacter{kCharacter.subCategory(1)};
  const TypeCategory kNonPlayerCharacter{kCharacter.subCategory(2)};
    const TypeCategory kMonster{kNonPlayerCharacter.subCategory(1)};
//       const TypeCategory kStandardMonster{kMonster.subCategory(1)};
      const TypeCategory kThiefMonster{kMonster.subCategory(2)};
      const TypeCategory kHunterMonster{kMonster.subCategory(3)};
//       const TypeCategory kQuestMonster{kMonster.subCategory(4)};
    const TypeCategory kCos{kNonPlayerCharacter.subCategory(3)};
      const TypeCategory kCart{kCos.subCategory(1)};
      const TypeCategory kTransport{kCos.subCategory(2)};
      const TypeCategory kSilkPet{kCos.subCategory(3)};
      const TypeCategory kGoldPet{kCos.subCategory(4)};
      const TypeCategory kMercenary{kCos.subCategory(5)};
      const TypeCategory kCaptured{kCos.subCategory(6)};
      const TypeCategory kFollower{kCos.subCategory(7)};
      const TypeCategory kAssaulter{kCos.subCategory(8)};
      const TypeCategory kPet2{kCos.subCategory(9)};
    const TypeCategory kSiegeObject{kNonPlayerCharacter.subCategory(4)};
    const TypeCategory kSiegeStruct{kNonPlayerCharacter.subCategory(5)};

const TypeCategory kItem{static_cast<uint8_t>(3)};
  const TypeCategory kEquipment{kItem.subCategory(1)};
//     const TypeCategory kGarment{kEquipment.subCategory(1)};
//     const TypeCategory kProtector{kEquipment.subCategory(2)};
//     const TypeCategory kArmor{kEquipment.subCategory(3)};
    const TypeCategory kWeapon{kEquipment.subCategory(6)};
//     const TypeCategory kSword{kWeapon.subCategory(2)};
//       const TypeCategory kOneHandedSword{kWeapon.subCategory(7)};
//       const TypeCategory kTwoHandedSword{kWeapon.subCategory(8)};
//       const TypeCategory kDualAxe{kWeapon.subCategory(9)};
    const TypeCategory kEquipmentExtra{kEquipment.subCategory(7)};
      const TypeCategory kTraderSuit{kEquipmentExtra.subCategory(1)};
      const TypeCategory kThiefSuit{kEquipmentExtra.subCategory(2)};
      const TypeCategory kHunterSuit{kEquipmentExtra.subCategory(3)};
  const TypeCategory kExpendable{kItem.subCategory(3)};
    const TypeCategory kRecoveryPotion{kExpendable.subCategory(1)};
      const TypeCategory kHpPotion{kRecoveryPotion.subCategory(1)};
      const TypeCategory kMpPotion{kRecoveryPotion.subCategory(2)};
      const TypeCategory kVigorPotion{kRecoveryPotion.subCategory(3)};
    const TypeCategory kCurePotion{kExpendable.subCategory(2)};
      const TypeCategory kPurificationPill{kCurePotion.subCategory(1)};
      const TypeCategory kUniversalPill{kCurePotion.subCategory(6)};
    const TypeCategory kScroll{kExpendable.subCategory(3)};
      const TypeCategory kReturnScroll{kScroll.subCategory(1)};
      const TypeCategory kReverseReturnScroll{kScroll.subCategory(3)};
    const TypeCategory kCurrency{kExpendable.subCategory(5)};
      const TypeCategory kGold{kCurrency.subCategory(0)};
    const TypeCategory kSpecialGoods{kExpendable.subCategory(8)};
    const TypeCategory kQuestAndEvent{kExpendable.subCategory(9)};
    const TypeCategory kAlchemyReinforce{kExpendable.subCategory(10)};
      const TypeCategory kElixir{kAlchemyReinforce.subCategory(1)};
      const TypeCategory kLuckyPowder{kAlchemyReinforce.subCategory(2)};
    const TypeCategory kSpecial{kExpendable.subCategory(13)};
      const TypeCategory kResurrection{kSpecial.subCategory(6)};

const TypeCategory kStructure{static_cast<uint8_t>(4)};

} // namespace type_id::categories