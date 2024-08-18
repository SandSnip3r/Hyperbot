#ifndef TYPE_ID_CATEGORIES_HPP_
#define TYPE_ID_CATEGORIES_HPP_

#include "typeCategory.hpp"

namespace type_id::categories {

// TODO: I think this can become a lot more manageable if TypeCategory was turned into a constexpr class.
// TODO: Provide a function to give a category name string from TypeId.

extern const TypeCategory kCharacter;
  extern const TypeCategory kPlayerCharacter;
  extern const TypeCategory kNonPlayerCharacter;
    extern const TypeCategory kMonster;
//       extern const TypeCategory kStandardMonster;
      extern const TypeCategory kThiefMonster;
      extern const TypeCategory kHunterMonster;
//       extern const TypeCategory kQuestMonster;
    extern const TypeCategory kCos;
      extern const TypeCategory kCart;
      extern const TypeCategory kTransport;
      extern const TypeCategory kSilkPet;
      extern const TypeCategory kGoldPet;
      extern const TypeCategory kMercenary;
      extern const TypeCategory kCaptured;
      extern const TypeCategory kFollower;
      extern const TypeCategory kAssaulter;
      extern const TypeCategory kPet2; // Also Siege (Ram) and Siege (Catapult)
    extern const TypeCategory kSiegeObject;
    extern const TypeCategory kSiegeStruct;

extern const TypeCategory kItem;
  extern const TypeCategory kEquipment;
//     extern const TypeCategory kGarment;
//     extern const TypeCategory kProtector;
//     extern const TypeCategory kArmor;
    extern const TypeCategory kShield;
      extern const TypeCategory kChineseShield;
      extern const TypeCategory kEuropeanShield;
    extern const TypeCategory kWeapon;
    extern const TypeCategory kEquipmentExtra;
      extern const TypeCategory kTraderSuit;
      extern const TypeCategory kThiefSuit;
      extern const TypeCategory kHunterSuit;
//     extern const TypeCategory kSword;
//       extern const TypeCategory kOneHandedSword;
//       extern const TypeCategory kTwoHandedSword;
//       extern const TypeCategory kDualAxe;
  extern const TypeCategory kExpendable;
    extern const TypeCategory kRecoveryPotion;
      extern const TypeCategory kHpPotion;
      extern const TypeCategory kMpPotion;
      extern const TypeCategory kVigorPotion;
    extern const TypeCategory kCurePotion;
      extern const TypeCategory kPurificationPill;
      extern const TypeCategory kUniversalPill;
    extern const TypeCategory kScroll;
      extern const TypeCategory kReturnScroll;
      extern const TypeCategory kReverseReturnScroll;
    extern const TypeCategory kAmmo;
    extern const TypeCategory kCurrency;
      extern const TypeCategory kGold;
    extern const TypeCategory kSpecialGoods;
    extern const TypeCategory kQuestAndEvent;
    extern const TypeCategory kAlchemyReinforce;
      extern const TypeCategory kElixir;
      extern const TypeCategory kLuckyPowder;
    extern const TypeCategory kSpecial;
      extern const TypeCategory kResurrection;

extern const TypeCategory kStructure;

} // namespace type_id::categories

#endif // TYPE_ID_CATEGORIES_HPP_