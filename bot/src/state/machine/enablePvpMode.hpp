#ifndef STATE_MACHINE_ENABLE_PVP_MODE_HPP_
#define STATE_MACHINE_ENABLE_PVP_MODE_HPP_

#include "event/event.hpp"
#include "stateMachine.hpp"

#include <string>

namespace state::machine {

class EnablePvpMode : public StateMachine {
public:
  EnablePvpMode(Bot &bot);
  ~EnablePvpMode() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"EnablePvpMode"};
  enum class State {
    kInit,
    kSentRequest,
    kCountdownRunning
  };
  State state_{State::kInit};
};

} // namespace state::machine

#endif // STATE_MACHINE_ENABLE_PVP_MODE_HPP_
