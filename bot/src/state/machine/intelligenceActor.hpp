#ifndef STATE_MACHINE_INTELLIGENCE_ACTOR_HPP_
#define STATE_MACHINE_INTELLIGENCE_ACTOR_HPP_

#include "common/pvpDescriptor.hpp"
#include "event/event.hpp"
#include "rl/ai/baseIntelligence.hpp"
#include "state/machine/stateMachine.hpp"

#include <silkroad_lib/pk2/ref/item.hpp>

#include <chrono>
#include <memory>
#include <optional>
#include <string>

namespace state::machine {

class IntelligenceActor : public StateMachine {
public:
  IntelligenceActor(StateMachine *parent, std::shared_ptr<rl::ai::BaseIntelligence> intelligence, common::PvpDescriptor::PvpId pvpId, sro::scalar_types::EntityGlobalId opponentGlobalId);
  ~IntelligenceActor() override;
  Status onUpdate(const event::Event *event) override;
protected:
  void injectPacket(const PacketContainer &packet, PacketContainer::Direction direction) override;
private:
  static inline std::chrono::milliseconds kPacketSendCooldown{200};
  static inline std::string kName{"IntelligenceActor"};
  std::shared_ptr<rl::ai::BaseIntelligence> intelligence_;
  const common::PvpDescriptor::PvpId pvpId_;
  const sro::scalar_types::EntityGlobalId opponentGlobalId_;
  std::optional<std::chrono::steady_clock::time_point> lastPacketTime_;

  bool isRelevantEvent(const event::Event *event) const;
};

} // namespace state::machine

#endif // STATE_MACHINE_INTELLIGENCE_ACTOR_HPP_
