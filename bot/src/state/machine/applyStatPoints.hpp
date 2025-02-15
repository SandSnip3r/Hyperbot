#ifndef STATE_MACHINE_APPLY_STAT_POINTS_HPP_
#define STATE_MACHINE_APPLY_STAT_POINTS_HPP_

#include "stateMachine.hpp"
#include "broker/eventBroker.hpp"

#include <optional>
#include <vector>

namespace state::machine {

enum class StatPointType {
  kInt,
  kStr
};

class ApplyStatPoints : public StateMachine {
public:
  ApplyStatPoints(Bot &bot, std::vector<StatPointType> statPointTypes);
  ~ApplyStatPoints() override;
  Status onUpdate(const event::Event *event) override;
private:
  static inline std::string kName{"ApplyStatPoints"};
  bool initialized_{false};
  std::vector<StatPointType> statPointTypes_;
  std::optional<uint16_t> lastInt_;
  std::optional<uint16_t> lastStr_;
  std::optional<broker::EventBroker::EventId> timeoutEventId_;
  Status initialize();
  void success();
};

} // namespace state::machine

#endif // STATE_MACHINE_APPLY_STAT_POINTS_HPP_