#include "talkingToShopNpc.hpp"
#include "talkingToStorageNpc.hpp"
#include "townlooping.hpp"
#include "useReturnScroll.hpp"
#include "walking.hpp"

#include "bot.hpp"
#include "logging.hpp"
#include "type_id/categories.hpp"

namespace state::machine {

Townlooping::Townlooping(Bot &bot) : StateMachine(bot) {
  stateMachineCreated(kName);
  // Build a shopping list
  // TODO: This should be based on a botting config
  shoppingList_ = {
    { 8, 5 }, //ITEM_ETC_HP_POTION_05 (XL hp potion)
    { 15, 5 }, //ITEM_ETC_MP_POTION_05 (XL mp potion)
    { 59, 50 }, //ITEM_ETC_CURE_ALL_05 (M special universal pill)
    { 10377, 50 }, //ITEM_ETC_CURE_RANDOM_04 (XL purification pill)
    // { 2198, 50 }, //ITEM_ETC_SCROLL_RETURN_02 (Special Return Scroll)
    // { 62, 1000 }, //ITEM_ETC_AMMO_ARROW_01 (Arrow)
    // { 3909, 1 }, //ITEM_COS_C_DHORSE1 (Ironclad Horse)
  };

  // Figure out which npcs we want to visit and in what order
  // TODO: This list should be constructed dynamically based on what we need to do in town
  npcsToVisit_ = { Npc::kStorage, Npc::kPotion , Npc::kGrocery, Npc::kBlacksmith, Npc::kProtector, Npc::kStable };

  if (npcsToVisit_.empty()) {
    LOG() << "No NPCs to visit in townloop" << std::endl;
    return;
  }

  if (!bot_.selfState().inTown()) {
    // Need to get to town
    const auto returnScrollSlots = bot_.selfState().inventory.findItemsOfCategory({type_id::categories::kReturnScroll});
    if (!returnScrollSlots.empty()) {
      // TODO: Make a decision of which to use; for now, we just use the first.
      setChildStateMachine<UseReturnScroll>(bot_, returnScrollSlots.front());
      return;
    } else {
      // No return scrolls, we must walk to town.
      // TODO: This will use a set of buffs for walking outside of town.
    }
  } else {
    // We are in town. Walk to the first NPC.
    // TODO: This will use a set of buffs for walking inside of town.
  }

  // For now, we just have one set of buffs for townlooping, no matter where we are
  // Check if we need to activate any of these buffs

  // Initialize state as walking to first NPC
  setChildStateMachine<Walking>(bot_, positionOfNpc(npcsToVisit_[currentNpcIndex_]));
}

Townlooping::~Townlooping() {
  stateMachineDestroyed();
}

void Townlooping::onUpdate(const event::Event *event) {
  if (done()) {
    return;
  }
  if (!childState_) {
    throw std::runtime_error("Not valid for childState_ to be empty");
  }
  childState_->onUpdate(event);
  if (childState_->done()) {
    // Figure out what to do next
    if (dynamic_cast<Walking*>(childState_.get()) != nullptr) {
      // Done walking, advance state
      if (npcsToVisit_[currentNpcIndex_] == Npc::kStorage) {
        setChildStateMachine<TalkingToStorageNpc>(bot_);
      } else {
        setChildStateMachine<TalkingToShopNpc>(bot_, npcsToVisit_[currentNpcIndex_], shoppingList_);
      }
    } else {
      // childState_ is UseReturnScroll, TalkingToStorageNpc, or TalkingToShopNpc
      if (dynamic_cast<UseReturnScroll*>(childState_.get()) == nullptr) {
        // childState_ is TalkingToStorageNpc or TalkingToShopNpc
        // We just finished with an npc, move to the next.
        ++currentNpcIndex_;
        if (done()) {
          // No more Npcs, done with townloop
          LOG() << "No more npcs to visit, done with townloop" << std::endl;
          return;
        }
      }

      // Update our state to walk to the next npc.
      setChildStateMachine<Walking>(bot_, positionOfNpc(npcsToVisit_[currentNpcIndex_]));
    }
    onUpdate(event);
    return;
  }
}

bool Townlooping::done() const {
  return (npcsToVisit_.empty() || currentNpcIndex_ == npcsToVisit_.size());
}

void Townlooping::buildBuffList() {
  // TODO: This data should come from some config
  // Create a list of buffs to use
  buffsToUse_ = std::vector<sro::scalar_types::ReferenceObjectId> {
      8150, // Ghost Walk - God
      // 8133, // God - Piercing Force
      // 7980, // Final Guard of Ice
      // 8183, // Concentration - 4th

      // Cleric
      // 11795, // Holy Recovery Division
      // 11934, // Holy Spell

      // Rogue
      // 9516, // Crossbow Extreme
    };

  // Move all buffs which require a weapon at all times to the end
  // TODO: Can do better, move skills which require a weapon at all times AND have a cooldown shorter than the skill duration even further to the back of the list
  // TODO: This exact function exists in Training also, move to a common area
  int backIndex = buffsToUse_.size()-1;
  for (int i=0; i<backIndex; ++i) {
    const auto &buffRefId = buffsToUse_[i];
    const auto &buffData = bot_.gameData().skillData().getSkillById(buffRefId);
    const auto buffRequiredWeapons = buffData.reqi();
    if (!buffRequiredWeapons.empty()) {
      // Buff requires a weapon at all times
      LOG() << "Moving buffs around. Moving " << i << " to " << backIndex << std::endl;
      std::swap(buffsToUse_[i], buffsToUse_[backIndex]);
      --backIndex;
    }
  }
}

std::optional<sro::scalar_types::ReferenceObjectId> Townlooping::getNextBuffToCast() const {
  // Lets evaluate our buffs and see if any need to be reactivated
  auto copyOfBuffsToUse = buffsToUse_;
  // Calculate a diff between what's active and what we want to be active
  for (const auto buff : bot_.selfState().buffs) {
    if (auto it = std::find(copyOfBuffsToUse.begin(), copyOfBuffsToUse.end(), buff); it != copyOfBuffsToUse.end()) {
      // This buff is already active, remove from list
      copyOfBuffsToUse.erase(it);
    }
  }

  if (copyOfBuffsToUse.empty()) {
    // No buffs need to be cast
    return {};
  }

  // Choose one that isnt active, on cooldown, or already used
  for (int i=0; i<copyOfBuffsToUse.size(); ++i) {
    const auto buff = copyOfBuffsToUse.at(i);
    if (bot_.selfState().alreadyTriedToCastSkill(buff)) {
      continue;
    }
    if (bot_.similarSkillIsAlreadyActive(buff)) {
      continue;
    }
    if (!bot_.canCastSkill(buff)) {
      // Cannot cast this skill right now. Maybe we're stunned, or it's on cooldown.
      continue;
    }
    return buff;
  }
  // Didn't find a buff that we need to cast.
  return {};
}

sro::Position Townlooping::positionOfNpc(Npc npc) const {
  // Hard code the Jangan NPCs' locations, for now
  static const auto npcPositionMap = []{
    std::map<Npc, sro::Position> npcPositions;
    npcPositions[Npc::kStorage]    = { 25000,  981.0f, 0.0f, 1032.0f };
    npcPositions[Npc::kPotion]     = { 25000, 1525.0f, 0.0f, 1385.0f };
    npcPositions[Npc::kGrocery]    = { 25000, 1618.0f, 0.0f, 1078.0f };
    npcPositions[Npc::kBlacksmith] = { 25000,  397.0f, 0.0f, 1358.0f };
    npcPositions[Npc::kProtector]  = { 25000,  363.0f, 0.0f, 1083.0f };
    npcPositions[Npc::kStable]     = { 25000,  390.0f, 0.0f,  493.0f };
    return npcPositions;
  }();

  const auto it = npcPositionMap.find(npc);
  if (it == npcPositionMap.end()) {
    throw std::runtime_error("Trying to get position of NPC which does not exist");
  }
  return it->second;
}

} // namespace state::machine