#ifndef STATE_MACHINE_PVP_AGENT_HPP_
#define STATE_MACHINE_PVP_AGENT_HPP_

#include "characterLoginInfo.hpp"
#include "event/event.hpp"
#include "state/machine/stateMachine.hpp"

#include <string>

namespace state::machine {

class PvpAgent : public StateMachine {
public:
  PvpAgent(Bot &bot, const CharacterLoginInfo &characterLoginInfo);
  ~PvpAgent() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"PvpAgent"};
  const CharacterLoginInfo characterLoginInfo_;
  Status initiatePvp(const event::BeginPvp &beginPvpEvent);
  Status startPvp(const event::Event *event);

  sro::scalar_types::EntityGlobalId getOpponentGlobalId();
  common::PvpDescriptor pvpDescriptor_;
  bool weAreReady_{false};
  bool opponentIsReady_{false};
};


} // namespace state::machine

#endif // STATE_MACHINE_PVP_AGENT_HPP_
