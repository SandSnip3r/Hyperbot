#ifndef STATE_MACHINE_STATEMACHINE_HPP_
#define STATE_MACHINE_STATEMACHINE_HPP_

#include "helpers.hpp"
#include "packet/structures/packetInnerStructures.hpp"

#include <map>
#include <variant>
#include <vector>

class Bot;
namespace event {
struct Event;
} // namespace event

namespace state::machine {

enum class Npc { kStorage, kPotion, kGrocery, kBlacksmith, kProtector, kStable };

class Walking {
public:
  Walking(Bot &bot, const std::vector<packet::structures::Position> &waypoints);
  void onUpdate(const event::Event *event);
  bool done() const;
private:
  Bot &bot_;
  std::vector<packet::structures::Position> waypoints_;
  size_t currentWaypointIndex_{0};
  bool requestedMovement_{false};
};

class TalkingToStorageNpc {
public:
  TalkingToStorageNpc(Bot &bot);
  void onUpdate(const event::Event *event);
  bool done() const;
private:
  Bot &bot_;
  // Hard coded npc global Id
  static constexpr const uint32_t kStorageNpcGId{0x000000CF};
  // Hard coded items to store
  static const uint16_t kArrowTypeId;
  static const uint16_t kHpPotionTypeId;

  enum class NpcInteractionState { kStart, kSelectionRequestPending, kNpcSelected, kStorageOpenRequestPending, kStorageOpened, kShopOpenRequestPending, kShopOpened, kDoneStoring };
  NpcInteractionState npcInteractionState_{NpcInteractionState::kStart};
  // start -> selectionRequestPending -> npcSelected -> storageOpenRequestPending -> storageOpened -> shopOpenRequestPending -> shopOpened
  //                                           |                                                               ^
  //                                           |_______________________________________________________________|

  bool pendingItemMovementRequest_{false};
  bool done_{false};

  void storeItems(const event::Event *event);
};

class TalkingToShopNpc {
public:
  TalkingToShopNpc(Bot &bot, Npc npc);
  void onUpdate(const event::Event *event);
  bool done() const;
private:
  Bot &bot_;
  Npc npc_;
  std::map<uint32_t, int> itemsToBuy_;
};

using TalkingToNpc = std::variant<std::monostate, TalkingToStorageNpc, TalkingToShopNpc>;

class Townlooping {
public:
  Townlooping(Bot &bot);
  ~Townlooping();
  void onUpdate(const event::Event *event);
  bool done() const;
private:
  Bot &bot_;
  std::vector<packet::Opcode> blockedOpcodes_;
  std::vector<Npc> npcsToVisit_;
  size_t currentNpcIndex_{0};
  std::variant<std::monostate, Walking, TalkingToNpc> childState_;

  void blockOpcode(packet::Opcode opcode);
  std::vector<packet::structures::Position> pathBetweenNpcs(Npc npcSrc, Npc npcDest) const;
};

} // namespace state::machine

std::ostream& operator<<(std::ostream &stream, state::machine::Npc npc);

#endif // STATE_MACHINE_STATEMACHINE_HPP_