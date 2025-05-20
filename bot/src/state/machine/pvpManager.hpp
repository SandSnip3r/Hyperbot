#ifndef STATE_MACHINE_PVP_MANAGER_HPP_
#define STATE_MACHINE_PVP_MANAGER_HPP_

#include "characterLoginInfo.hpp"
#include "event/event.hpp"
#include "state/machine/stateMachine.hpp"

#include <optional>
#include <string>

namespace state::machine {

/** This state machine sits idly until a pvp assignment arrives.
  * If an assignment arrives for the controlled character, this state machine prepares for pvp then pvps.
***/
class PvpManager : public StateMachine {
public:
  PvpManager(Bot &bot);
  ~PvpManager() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"PvpManager"};
  Status startPvp(const event::Event *event);
  void resetAndNotifyReadyForAssignment();
  void setPrepareForPvpStateMachine();

  bool initialized_{false};
  std::optional<sro::scalar_types::EntityGlobalId> opponentGlobalId_;
  std::optional<common::PvpDescriptor> pvpDescriptor_;
  bool publishedThatWeAreReadyForAssignment_{false};
  bool weAreReady_{false};
  bool opponentIsReady_{false};
  bool receivedResurrectionOption_{false};

  bool isPvping() const;
};


} // namespace state::machine

#endif // STATE_MACHINE_PVP_MANAGER_HPP_
