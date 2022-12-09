#ifndef STATE_MACHINE_TALKING_TO_STORAGE_NPC_HPP_
#define STATE_MACHINE_TALKING_TO_STORAGE_NPC_HPP_

#include "stateMachine.hpp"

#include <cstdint>
#include <set>

namespace state::machine {

class TalkingToStorageNpc : public StateMachine {
public:
  TalkingToStorageNpc(Bot &bot);
  ~TalkingToStorageNpc() override;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  static inline std::string kName{"TalkingToStorageNpc"};
  // Hard coded npc global Id
  static constexpr const uint32_t kStorageNpcGId{0x000000CF};
  std::set<uint16_t> itemTypesToStore_;

  enum class NpcInteractionState { kStart, kSelectionRequestPending, kNpcSelected, kStorageOpenRequestPending, kStorageOpened, kShopOpenRequestPending, kShopOpened, kDoneStoring };
  NpcInteractionState npcInteractionState_{NpcInteractionState::kStart};
  // start -> selectionRequestPending -> npcSelected -> storageOpenRequestPending -> storageOpened -> shopOpenRequestPending -> shopOpened
  //                                           |                                                               ^
  //                                           |_______________________________________________________________|

  bool pendingItemMovementRequest_{false};
  bool done_{false};

  void storeItems(const event::Event *event);
};

} // namespace state::machine

#endif // STATE_MACHINE_TALKING_TO_STORAGE_NPC_HPP_