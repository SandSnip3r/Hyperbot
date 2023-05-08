#include "castSkill.hpp"
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
  buildBuffList();
  buildShoppingList();
  buildNpcList();
}

Townlooping::~Townlooping() {
  stateMachineDestroyed();
}

void Townlooping::onUpdate(const event::Event *event) {
  if (done()) {
    return;
  }

  if (childState_) {
    childState_->onUpdate(event);
    if (!childState_->done()) {
      // Child still running, nothing to do.
      return;
    }
    // Child state is done, figure out what to do next.
    if (dynamic_cast<Walking*>(childState_.get()) != nullptr) {
      // Done walking, update state to have us now talk to the NPC we just arrived at.
      // Doing it this way means that there's no opportunity to cast buffs before talking to the NPC, but I think that's ok.
      if (npcsToVisit_[currentNpcIndex_] == Npc::kStorage) {
        setChildStateMachine<TalkingToStorageNpc>(bot_);
      } else {
        setChildStateMachine<TalkingToShopNpc>(bot_, npcsToVisit_[currentNpcIndex_], shoppingList_);
      }
      onUpdate(event);
      return;
    } else if (dynamic_cast<TalkingToStorageNpc*>(childState_.get()) != nullptr ||
               dynamic_cast<TalkingToShopNpc*>(childState_.get()) != nullptr) {
      // We just finished with an npc, advance our state.
      ++currentNpcIndex_;
      if (done()) {
        // No more Npcs, done with townloop
        LOG() << "No more npcs to visit, done with townloop" << std::endl;
        return;
      }
      // Update our state to walk to the next npc.
      setChildStateMachine<Walking>(bot_, positionOfNpc(npcsToVisit_[currentNpcIndex_]));
      onUpdate(event);
      return;
    } else if (dynamic_cast<UseReturnScroll*>(childState_.get()) != nullptr) {
      // Finished using a return scroll
      waitingForSpawn_ = true;
    } else {
      throw std::runtime_error("Unknown child state type");
    }
    childState_.reset();
  }

  if (event != nullptr) {
    // Originally, I thought that once we received the Character Data packet, we were spawned and could begin to act, but It seems like we cannot do anything for a little time after that. The server wont even ack our sent packets. Instead, it seems that we consistently get a state update packet shortly after spawning. We'll use that as an indication that we've spawned and can begin taking actions. For non-GM characters, the first body state is BodyState::kUntouchable.
    if (const auto *bodyStateChangedEvent = dynamic_cast<const event::EntityBodyStateChanged*>(event)) {
      if (bodyStateChangedEvent->globalId == bot_.selfState().globalId) {
        waitingForSpawn_ = false;
      }
    }
  }

  if (waitingForSpawn_) {
    return;
  }

  // First, check if we're out of town and have a return scroll to use
  if (!bot_.worldState().selfState().inTown()) {
    // Not in town
    const auto returnScrollSlots = bot_.selfState().inventory.findItemsOfCategory({type_id::categories::kReturnScroll});
    if (!returnScrollSlots.empty()) {
      if (sanityCheckUsedReturnScroll_) {
        throw std::runtime_error("We already used a return scroll and want to use another. Something is wrong");
      }
      // TODO: Make a decision of which to use; for now, we just use the first.
      setChildStateMachine<UseReturnScroll>(bot_, returnScrollSlots.front());
      onUpdate(event);
      sanityCheckUsedReturnScroll_ = true;
      return;
    }
  }

  // We're either in town, or out of town and must walk to town. In either case, make sure we're buffed up, and walk to the next NPC.
  // First, try to cast buffs.
  const auto nextBuffToCast = getNextBuffToCast();
  if (nextBuffToCast) {
    auto castSkillBuilder = CastSkillStateMachineBuilder(bot_, *nextBuffToCast);
    const auto &buffData = bot_.gameData().skillData().getSkillById(*nextBuffToCast);

    // Does the buff require a specific weapon to be equipped to cast?
    const auto weaponSlot = getInventorySlotOfWeaponForSkill(buffData, bot_);
    // Note: It is also possible that a skill requires a shield (shield bash)
    if (weaponSlot) {
      castSkillBuilder.withWeapon(*weaponSlot);
    }

    // TODO: Shield too?

    if (buffData.targetRequired) {
      // TODO: We assume this buff is for ourself
      castSkillBuilder.withTarget(bot_.selfState().globalId);
    }

    setChildStateMachine(castSkillBuilder.create());
    onUpdate(event);
    return;
  }

  // Done with buffs, walk to next NPC.
  setChildStateMachine<Walking>(bot_, positionOfNpc(npcsToVisit_[currentNpcIndex_]));
  onUpdate(event);
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
      std::swap(buffsToUse_[i], buffsToUse_[backIndex]);
      --backIndex;
    }
  }
}

void Townlooping::buildShoppingList() {
  // TODO: This should be based on a botting config
  shoppingList_ = {
    { 8, 5 }, //ITEM_ETC_HP_POTION_05 (XL hp potion)
    { 15, 100 }, //ITEM_ETC_MP_POTION_05 (XL mp potion)
    { 59, 50 }, //ITEM_ETC_CURE_ALL_05 (M special universal pill)
    { 10377, 50 }, //ITEM_ETC_CURE_RANDOM_04 (XL purification pill)
    { 2198, 1 }, //ITEM_ETC_SCROLL_RETURN_02 (Special Return Scroll)
    // { 62, 1000 }, //ITEM_ETC_AMMO_ARROW_01 (Arrow)
    // { 3909, 1 }, //ITEM_COS_C_DHORSE1 (Ironclad Horse)
  };
}

void Townlooping::buildNpcList() {
  // Figure out which npcs we want to visit and in what order
  // TODO: This list should be constructed dynamically based on what we need to do in town
  npcsToVisit_ = { Npc::kStorage, Npc::kPotion , Npc::kGrocery, Npc::kBlacksmith, Npc::kProtector, Npc::kStable };
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
    if (bot_.selfState().skillEngine.alreadyTriedToCastSkill(buff)) {
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