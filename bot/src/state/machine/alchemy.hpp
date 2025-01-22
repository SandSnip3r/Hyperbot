#ifndef STATE_MACHINE_ALCHEMY_HPP_
#define STATE_MACHINE_ALCHEMY_HPP_

#include "stateMachine.hpp"

#include "broker/eventBroker.hpp"

#include <silkroad_lib/scalar_types.hpp>
#include <silkroad_lib/position.hpp>

#include <optional>

namespace state::machine {

class Alchemy : public StateMachine {
public:
  Alchemy(Bot &bot);
  ~Alchemy() override;
  void onUpdate(const event::Event *event) override;
  bool done() const override;
private:
  static constexpr const sro::scalar_types::OptLevelType goalOptLevel_ = 200;
  static constexpr const int kAlchemyTimedOutMs{5000};
  static constexpr const int kMakeItemTimedOutMs{5000};
  static constexpr const bool kUseLuckStones{false};
  static inline std::string kName{"Alchemy"};
  bool done_{false};
  sro::Position startPosition_;
  bool waitingForCreatedElixir_{false};
  bool waitingForCreatedPowder_{false};
  bool waitingForCreatedBlade_{false};
  int nextBladePlusToSpawn_{1};
  std::optional<sro::scalar_types::ReferenceObjectId> waitingForCreatedItem_;
  std::optional<broker::EventBroker::EventId> alchemyTimedOutEventId_;
  std::optional<broker::EventBroker::EventId> makeItemTimedOutEventId_;

  void initialize();
};

} // namespace state::machine

#endif // STATE_MACHINE_ALCHEMY_HPP_