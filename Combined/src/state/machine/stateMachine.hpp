#ifndef STATE_MACHINE_STATEMACHINE_HPP_
#define STATE_MACHINE_STATEMACHINE_HPP_

#include "helpers.hpp"
#include "packet/opcode.hpp"
#include "packet/structures/packetInnerStructures.hpp"

#include <silkroad_lib/position.h>

#include <map>
#include <memory>
#include <vector>

class Bot;
namespace event {
struct Event;
} // namespace event

namespace state::machine {

enum class Npc { kStorage, kPotion, kGrocery, kBlacksmith, kProtector, kStable };

class StateMachine {
public:
  StateMachine(Bot &bot);
  virtual ~StateMachine();
  virtual void onUpdate(const event::Event *event) = 0;
  virtual bool done() const = 0; // TODO: rename to isDone
protected:
  Bot &bot_;
  void pushBlockedOpcode(packet::Opcode opcode);
private:
  std::vector<packet::Opcode> blockedOpcodes_;
};

class Walking : public StateMachine {
public:
  Walking(Bot &bot, const sro::Position &destinationPosition);
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  std::vector<sro::Position> waypoints_;
  size_t currentWaypointIndex_{0};
  bool requestedMovement_{false};
  std::vector<sro::Position> calculatePathToDestination(const sro::Position &destinationPosition) const;
};

class BuyingItems : public StateMachine {
public:
  struct PurchaseRequest {
    uint8_t tabIndex;
    uint8_t itemIndex;
    uint16_t quantity;
    int32_t maxStackSize;
  };
  BuyingItems(Bot &bot, const std::map<uint32_t, PurchaseRequest> &itemsToBuy);
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  std::map<uint32_t, PurchaseRequest> itemsToBuy_;
  bool waitingOnBuyResponse_{false};
  bool waitingOnItemMovementResponse_{false};
  bool done_{false};
};

class TalkingToStorageNpc : public StateMachine {
public:
  using StateMachine::StateMachine;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
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

class TalkingToShopNpc : public StateMachine {
public:
  TalkingToShopNpc(Bot &bot, Npc npc, const std::map<uint32_t, int> &shoppingList);
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  Npc npc_;
  const std::map<uint32_t, int> &shoppingList_;
  uint32_t npcGid_;
  std::map<uint32_t, BuyingItems::PurchaseRequest> itemsToBuy_;
  std::unique_ptr<BuyingItems> buyingItemsChildState_;
  bool waitingForSelectionResponse_{false};
  bool waitingForTalkResponse_{false};
  bool waitingForRepairResponse_{false};
  bool waitingOnStopTalkResponse_{false};
  bool waitingOnDeselectionResponse_{false};
  bool done_{false};

  void figureOutWhatToBuy();
  bool needToRepair() const;
  bool doneBuyingItems() const;
  bool doneWithNpc() const;
};

class Townlooping : public StateMachine {
public:
  Townlooping(Bot &bot);
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  std::map<uint32_t, int> shoppingList_;
  std::vector<Npc> npcsToVisit_;
  size_t currentNpcIndex_{0};
  std::unique_ptr<StateMachine> childState_;

  sro::Position positionOfNpc(Npc npc) const;
};

class Training : public StateMachine {
public:
  Training(Bot &bot, const sro::Position &trainingSpotCenter);
  ~Training() override;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  sro::Position trainingSpotCenter_;
  static constexpr double kMonsterRange_{250};
  static constexpr double kItemRange_{300};
};

class Botting : public StateMachine {
public:
  Botting(Bot &bot);
  void onUpdate(const event::Event *event) override;
  bool done() const override;
  void initialize();
  void reset();
private:
  std::unique_ptr<StateMachine> childState_;
  sro::Position trainingSpotCenter_;
};

} // namespace state::machine

std::ostream& operator<<(std::ostream &stream, state::machine::Npc npc);

#endif // STATE_MACHINE_STATEMACHINE_HPP_