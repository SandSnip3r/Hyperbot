#ifndef STATE_MACHINE_APPLY_STAT_POINTS_HPP_
#define STATE_MACHINE_APPLY_STAT_POINTS_HPP_

#include "stateMachine.hpp"

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
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  static inline std::string kName{"ApplyStatPoints"};
  std::vector<StatPointType> statPointTypes_;
  std::optional<uint16_t> lastInt_;
  std::optional<uint16_t> lastStr_;
  bool waiting_{false};
  bool done_{false};
};

} // namespace state::machine

#endif // STATE_MACHINE_APPLY_STAT_POINTS_HPP_