#ifndef STATE_MACHINE_ENSURE_FULL_VITALS_AND_NO_STATUSES_HPP_
#define STATE_MACHINE_ENSURE_FULL_VITALS_AND_NO_STATUSES_HPP_

#include "broker/eventBroker.hpp"
#include "event/event.hpp"
#include "state/machine/stateMachine.hpp"

#include <silkroad_lib/scalar_types.hpp>

#include <optional>
#include <string>

namespace state::machine {

class EnsureFullVitalsAndNoStatuses : public StateMachine {
public:
  EnsureFullVitalsAndNoStatuses(StateMachine *parent);
  ~EnsureFullVitalsAndNoStatuses() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"EnsureFullVitalsAndNoStatuses"};
  sro::scalar_types::ReferenceObjectId vigorPotionItemId_;
  sro::scalar_types::ReferenceObjectId universalPillItemId_;
  sro::scalar_types::ReferenceObjectId purificationPillItemId_;
  std::optional<broker::EventBroker::EventId> waitForPotionEventId_;
  };

} // namespace state::machine

#endif // STATE_MACHINE_ENSURE_FULL_VITALS_AND_NO_STATUSES_HPP_
