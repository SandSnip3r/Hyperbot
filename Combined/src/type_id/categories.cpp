#include "categories.hpp"

namespace type_id::categories {

// const TypeCategory kCharacter{1};
//   const TypeCategory kPlayerCharacter{kCharacter.subCategory(1)};
//   const TypeCategory kNonPlayerCharacter{kCharacter.subCategory(2)};
//     const TypeCategory kMonster{kNonPlayerCharacter.subCategory(1)};
//       const TypeCategory kStandardMonster{kMonster.subCategory(1)};
//       const TypeCategory kQuestMonster{kMonster.subCategory(4)};

const TypeCategory kItem{3};
  const TypeCategory kEquipment{kItem.subCategory(1)};
//     const TypeCategory kGarment{kEquipment.subCategory(1)};
//     const TypeCategory kProtector{kEquipment.subCategory(2)};
//     const TypeCategory kArmor{kEquipment.subCategory(3)};
    const TypeCategory kWeapon{kEquipment.subCategory(6)};
//     const TypeCategory kSword{kWeapon.subCategory(2)};
//       const TypeCategory kOneHandedSword{kWeapon.subCategory(7)};
//       const TypeCategory kTwoHandedSword{kWeapon.subCategory(8)};
//       const TypeCategory kDualAxe{kWeapon.subCategory(9)};
  const TypeCategory kExpendable{kItem.subCategory(3)};
    const TypeCategory kRecoveryPotion{kExpendable.subCategory(1)};
      const TypeCategory kHpPotion{kRecoveryPotion.subCategory(1)};
      const TypeCategory kMpPotion{kRecoveryPotion.subCategory(2)};
      const TypeCategory kVigorPotion{kRecoveryPotion.subCategory(3)};
//     const TypeCategory kCurePotion{kExpendable.subCategory(2)};
//       const TypeCategory kPurificationPill{kCurePotion.subCategory(1)};
//       const TypeCategory kUniversalPill{kCurePotion.subCategory(7)};

} // namespace type_id::categories