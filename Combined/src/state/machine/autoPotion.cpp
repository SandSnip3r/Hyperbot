#include "autoPotion.hpp"
#include "useItem.hpp"

#include "bot.hpp"
#include "event/event.hpp"
#include "logging.hpp"
#include "helpers.hpp"
#include "type_id/categories.hpp"

namespace state::machine {

AutoPotion::AutoPotion(Bot &bot) : StateMachine(bot) {
}

void AutoPotion::onUpdate(const event::Event *event) {
  // First, prefer to defer to child state
  if (childState_) {
    childState_->onUpdate(event);
    if (childState_->done()) {
      // Must be done using an item
      childState_.reset();
    } else {
      // Still using an item, nothing to do
      return;
    }
  }

  // We don't care about any events at the moment

  if (!bot_.selfState().canUseItems()) {
    // Nothing to do if we can't use items
    return;
  }

  using UseFunction = std::function<bool()>;
  // This ordering is kind of arbitrary, but it will dictate a priority since we only use one item at a time
  std::vector<UseFunction> pillsAndPotionUseFunctions = {
    std::bind(&AutoPotion::tryUsePurificationPill, this),
    std::bind(&AutoPotion::tryUseHpPotion, this),
    std::bind(&AutoPotion::tryUseMpPotion, this),
    std::bind(&AutoPotion::tryUseUniversalPill, this)
  };

  bool usedAnItem{false};
  for (auto &useFunction : pillsAndPotionUseFunctions) {
    usedAnItem = useFunction();
    if (usedAnItem) {
      // Only use one item at a time
      break;
    }
  }

  if (usedAnItem) {
    // Child State has been set, recurse
    onUpdate(event);
  }
}

bool AutoPotion::done() const {
  // Autopotion is never done
  return false;
}

bool AutoPotion::tryUsePurificationPill() {
  const auto modernStateLevels = bot_.selfState().modernStateLevels();
  if (!std::any_of(modernStateLevels.begin(), modernStateLevels.end(), [](auto level){ return level > 0; })) {
    // Don't have any modern statuses. Don't need to use a purification pill.
    return false;
  }

  if (!bot_.selfState().canUseItem(type_id::categories::kPurificationPill)) {
    return false;
  }

  uint8_t currentCureLevel = 0;
  std::set<int32_t> processedCureLevels;
  sro::scalar_types::StorageIndexType bestOptionSlotNum;
  const auto &inventory = bot_.selfState().inventory;

  for (uint8_t slotNum=0; slotNum<inventory.size(); ++slotNum) {
    if (!inventory.hasItem(slotNum)) {
      continue;
    }
    const storage::Item *item = inventory.getItem(slotNum);
    if (!type_id::categories::kPurificationPill.contains(item->typeData())) {
      // Item is not a purification pill
      continue;
    }
    const auto *itemAsExpendable = dynamic_cast<const storage::ItemExpendable*>(item);
    if (itemAsExpendable == nullptr) {
      throw std::runtime_error("Item is a purification pill but is not an expendable");
    }
    if (itemAsExpendable->quantity == 0) {
      // Last one was used in this stack, skip
      continue;
    }
    const auto pillCureStateBitmask = itemAsExpendable->itemInfo->param1;
    const auto curableStatesWeHave = (pillCureStateBitmask & bot_.selfState().stateBitmask());
    if (curableStatesWeHave == 0) {
      // This pill cannot cure any of the type(s) of statuses that we have
      continue;
    }
    // pillTreatmentLevel is the max status level that the pill will treat (corresponding to degree; eg 1-12)
    const auto pillTreatmentLevel = item->itemInfo->param2;
    if (processedCureLevels.find(pillTreatmentLevel) != processedCureLevels.end()) {
      // We've already processed a pill which cures at this level
      continue;
    }

    // Figure out what our highest level state is, this will determine which pill we use
    uint8_t maxStateLevel = 0;
    for (uint32_t bitNum=0; bitNum<32; ++bitNum) {
      const auto bit = 1 << bitNum;
      if (curableStatesWeHave & bit) {
        maxStateLevel = std::max(maxStateLevel, modernStateLevels[bitNum]);
      }
    }

    // Choose the biggest pill which isn't wasteful
    if (pillTreatmentLevel > currentCureLevel) {
      // Found a larger pill than we've seen before (or our first pill)
      // Was the previous pill already larger than our worst status?
      if (currentCureLevel >= maxStateLevel) {
        // The new pill is overkill, the last one was big enough
      } else {
        // The previous pill wasn't large enough, lets use this one instead
        bestOptionSlotNum = slotNum;
        currentCureLevel = pillTreatmentLevel;
        processedCureLevels.emplace(currentCureLevel);
      }
    } else {
      // Is a smaller pill, maybe it's more efficient?
      if (pillTreatmentLevel >= maxStateLevel) {
        // Is enough, choose this one instead
        bestOptionSlotNum = slotNum;
        currentCureLevel = pillTreatmentLevel;
        processedCureLevels.emplace(currentCureLevel);
      }
    }
  }

  if (currentCureLevel != 0) {
    // Found the a pill that will cure something
    useItem(bestOptionSlotNum);
    return true;
  }
  return false;
}

bool AutoPotion::tryUseHpPotion() {
  if (!bot_.selfState().maxHp()) {
    // Don't yet know our max hp
    return false;
  }
  const double hpPercentage = static_cast<double>(bot_.selfState().currentHp()) / *bot_.selfState().maxHp();

  const auto legacyStateEffects = bot_.selfState().legacyStateEffects();
  const bool haveZombie = (legacyStateEffects[helpers::toBitNum(packet::enums::AbnormalStateFlag::kZombie)] > 0);
  if (haveZombie) {
    // Don't want to use a health or vigor potion, there is nothing else to do in this function
    return false;
  }

  bool usedAnItem = false;
  if (hpPercentage < bot_.currentCharacterConfig().autopotion_config().hp_threshold() && bot_.selfState().canUseItem(type_id::categories::kHpPotion)) {
    // Use health potion
    usedAnItem = usePotion(type_id::categories::kHpPotion);
  }

  if (usedAnItem) {
    // Don't use anything else at the same time
    return true;
  }

  if (hpPercentage < bot_.currentCharacterConfig().autopotion_config().vigor_hp_threshold() &&
      bot_.selfState().canUseItem(type_id::categories::kVigorPotion)) {
    // Use vigor potion
    usedAnItem = usePotion(type_id::categories::kVigorPotion);
  }
  return usedAnItem;
}

bool AutoPotion::tryUseMpPotion() {
  if (!bot_.selfState().maxMp()) {
    // Don't yet know our max mp
    return false;
  }
  const double mpPercentage = static_cast<double>(bot_.selfState().currentMp()) / *bot_.selfState().maxMp();

  bool usedAnItem = false;
  if (mpPercentage < bot_.currentCharacterConfig().autopotion_config().mp_threshold() && bot_.selfState().canUseItem(type_id::categories::kMpPotion)) {
    // Use mana potion
    usedAnItem = usePotion(type_id::categories::kMpPotion);
  }

  if (usedAnItem) {
    // Don't use anything else at the same time
    return true;
  }

  const auto legacyStateEffects = bot_.selfState().legacyStateEffects();
  const bool haveZombie = (legacyStateEffects[helpers::toBitNum(packet::enums::AbnormalStateFlag::kZombie)] > 0);
  if (haveZombie) {
    // Don't use vigors when we have zombie
    return false;
  }

  if (mpPercentage < bot_.currentCharacterConfig().autopotion_config().vigor_mp_threshold() &&
      bot_.selfState().canUseItem(type_id::categories::kVigorPotion)) {
    // Use vigor potion
    usedAnItem = usePotion(type_id::categories::kVigorPotion);
  }
  return usedAnItem;
}

bool AutoPotion::tryUseUniversalPill() {
  const auto legacyStateEffects = bot_.selfState().legacyStateEffects();
  if (!std::any_of(legacyStateEffects.begin(), legacyStateEffects.end(), [](const uint16_t effect){ return effect > 0; })) {
    // Don't have any legacy statuses. Don't need to use a universal pill.
    return false;
  }

  if (!bot_.selfState().canUseItem(type_id::categories::kUniversalPill)) {
    return false;
  }

  // Figure out our status with the highest effect
  uint16_t ourWorstStatusEffect = *std::max_element(legacyStateEffects.begin(), legacyStateEffects.end());
  int32_t bestCure = 0;
  uint8_t bestOptionSlotNum;
  const auto &inventory = bot_.selfState().inventory;

  for (uint8_t slotNum=0; slotNum<inventory.size(); ++slotNum) {
    if (!inventory.hasItem(slotNum)) {
      continue;
    }
    const storage::Item *item = inventory.getItem(slotNum);
    if (!type_id::categories::kUniversalPill.contains(item->typeData())) {
      // Item is not a universal pill
      continue;
    }
    const auto *itemAsExpendable = dynamic_cast<const storage::ItemExpendable*>(item);
    if (itemAsExpendable == nullptr) {
      throw std::runtime_error("Item is a universal pill but is not an expendable");
    }
    if (bestCure == 0) {
      // First pill found, at least we can use this
      bestCure = itemAsExpendable->itemInfo->param1;
      bestOptionSlotNum = slotNum;
    } else {
      // Already have a choice, lets see if this is better
      const auto thisPillCureEffect = itemAsExpendable->itemInfo->param1;
      const bool curesEverything = (thisPillCureEffect >= ourWorstStatusEffect);
      const bool curesMoreThanPrevious = (thisPillCureEffect >= ourWorstStatusEffect && bestCure < ourWorstStatusEffect);
      if (curesEverything && thisPillCureEffect < bestCure) {
        // Found a smaller pill that can cure everything
        bestCure = thisPillCureEffect;
        bestOptionSlotNum = slotNum;
      } else if (curesMoreThanPrevious && thisPillCureEffect > bestCure) {
        // Found a pill that can cure more without being wasteful
        bestCure = thisPillCureEffect;
        bestOptionSlotNum = slotNum;
      }
    }
  }
  if (bestCure != 0) {
    // Found the best pill
    useItem(bestOptionSlotNum);
    return true;
  }
  return false;
}

bool AutoPotion::usePotion(const type_id::TypeCategory &potionType) {
  // Find the first one in our inventory
  const auto &inventory = bot_.selfState().inventory;

  for (uint8_t slotNum=0; slotNum<inventory.size(); ++slotNum) {
    if (!inventory.hasItem(slotNum)) {
      continue;
    }
    const storage::Item *item = inventory.getItem(slotNum);
    if (!potionType.contains(item->typeData())) {
      // This is not the desired potion type
      continue;
    }

    useItem(slotNum);
    return true;
  }

  // We didn't have this type of potion
  return false;
}

void AutoPotion::useItem(sro::scalar_types::StorageIndexType itemIndex) {
  if (childState_) {
    throw std::runtime_error("Trying to use an item, but we already have a child state");
  }
  setChildStateMachine<UseItem>(itemIndex);
}

} // namespace state::machine