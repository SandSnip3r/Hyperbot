#ifndef STATE_MACHINE_ENABLE_PVP_MODE_HPP_
#define STATE_MACHINE_ENABLE_PVP_MODE_HPP_

#include "broker/eventBroker.hpp"
#include "event/event.hpp"
#include "stateMachine.hpp"

#include <optional>
#include <string>

namespace state::machine {

class EnablePvpMode : public StateMachine {
public:
  EnablePvpMode(StateMachine *parent);
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
  std::optional<broker::EventBroker::EventId> requestTimeoutEventId_;
  void sendRequest();
};

} // namespace state::machine

#endif // STATE_MACHINE_ENABLE_PVP_MODE_HPP_
