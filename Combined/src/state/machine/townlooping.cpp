#include "talkingToShopNpc.hpp"
#include "talkingToStorageNpc.hpp"
#include "townlooping.hpp"
#include "walking.hpp"

#include "logging.hpp"

namespace state::machine {

Townlooping::Townlooping(Bot &bot) : StateMachine(bot) {
  // Build a shopping list
  // TODO: This should be based on a botting config
  shoppingList_ = {
    { 8, 50 }, //ITEM_ETC_HP_POTION_05 (XL hp potion)
    { 15, 200 }, //ITEM_ETC_MP_POTION_05 (XL mp potion)
    { 59, 100 }, //ITEM_ETC_CURE_ALL_05 (M special universal pill)
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

  // Initialize state as walking to first NPC
  childState_ = std::make_unique<Walking>(bot_, positionOfNpc(npcsToVisit_[currentNpcIndex_]));
}

void Townlooping::onUpdate(const event::Event *event) {
TODO_REMOVE_THIS_LABEL:
  if (done()) {
    return;
  }
  if (auto *walkingState = dynamic_cast<Walking*>(childState_.get())) {
    walkingState->onUpdate(event);

    if (walkingState->done()) {
      // Done walking, advance state
      if (npcsToVisit_[currentNpcIndex_] == Npc::kStorage) {
        childState_ = std::make_unique<TalkingToStorageNpc>(bot_);
      } else {
        childState_ = std::make_unique<TalkingToShopNpc>(bot_, npcsToVisit_[currentNpcIndex_], shoppingList_);
      }
      // TODO: Go back to the top of this function
      goto TODO_REMOVE_THIS_LABEL;
    }
  } else {
    if (!childState_) {
      throw std::runtime_error("Not valid for childState_ to be empty");
    }
    // childState_ is either TalkingToStorageNpc or TalkingToShopNpc
    childState_->onUpdate(event);
    if (childState_->done()) {
      // Moving on to next npc
      ++currentNpcIndex_;
      if (done()) {
        // No more Npcs, done with townloop
        LOG() << "No more npcs to visit, done with townloop" << std::endl;
        return;
      }

      // Update our state to walk to the next npc
      childState_ = std::make_unique<Walking>(bot_, positionOfNpc(npcsToVisit_[currentNpcIndex_]));
      // TODO: Go back to the top of this function
      goto TODO_REMOVE_THIS_LABEL;
    }
  }
}

bool Townlooping::done() const {
  return (npcsToVisit_.empty() || currentNpcIndex_ == npcsToVisit_.size());
}

sro::Position Townlooping::positionOfNpc(Npc npc) const {
  // Hard code the NPCs' locations, for now
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