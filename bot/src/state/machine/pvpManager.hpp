#ifndef STATE_MACHINE_PVP_MANAGER_HPP_
#define STATE_MACHINE_PVP_MANAGER_HPP_

#include "characterLoginInfo.hpp"
#include "event/event.hpp"
#include "state/machine/stateMachine.hpp"

#include <optional>
#include <string>

namespace state::machine {

class PvpManager : public StateMachine {
public:
  PvpManager(Bot &bot, const CharacterLoginInfo &characterLoginInfo);
  ~PvpManager() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"PvpManager"};
  const CharacterLoginInfo characterLoginInfo_;
  Status startPvp(const event::Event *event);
  void resetAndNotifyReadyForAssignment();
  void setPrepareForPvpStateMachine();

  sro::scalar_types::EntityGlobalId getOpponentGlobalId();
  bool initialized_{false};
  std::optional<common::PvpDescriptor> pvpDescriptor_;
  bool publishedThatWeAreReadyForAssignment_{false};
  bool weAreReady_{false};
  bool opponentIsReady_{false};
  bool receivedResurrectionOption_{false};
};


} // namespace state::machine

#endif // STATE_MACHINE_PVP_MANAGER_HPP_
