#ifndef ENTITY_MOBILE_ENTITY_HPP_
#define ENTITY_MOBILE_ENTITY_HPP_

#include "entity.hpp"
#include "geometry.hpp"
#include "broker/eventBroker.hpp"

#include "shared/silkroad_security.h"

#include <silkroad_lib/position.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <ostream>
#include <optional>

namespace entity {

enum class MotionState : uint8_t {
  kStand = 0,
  kSkill = 1,
  kWalk = 2,
  kRun = 3,
  kSit = 4
  // kJump = 5,
  // kSwim = 6,
  // kRide = 7,
  // kKnockdown = 8,
  // kStun = 9,
  // kFrozen = 10,
  // kHit = 11,
  // kReqHelp = 12,
  // kPao = 13,
  // kCounterattack = 14,
  // kSkillActionOff = 15,
  // kSkillKnockback = 16,
  // kSkillProtectionWall = 17,
  // kChangeMotion = 18,
};

class MobileEntity : public Entity {
public:
  ~MobileEntity() override;
  std::chrono::steady_clock::time_point startedMovingTime;
  std::optional<sro::Position> destinationPosition;
  MotionState motionState;
  std::optional<MotionState> lastMotionState;
  float walkSpeed;
  float runSpeed;
  std::optional<broker::EventBroker::EventId> movingEventId;
  void initializeAsMoving(const sro::Position &destinationPosition, const PacketContainer::Clock::time_point &timestamp);
  void initializeAsMoving(sro::Angle destinationAngle, const PacketContainer::Clock::time_point &timestamp);
  void initializeEventBroker(broker::EventBroker &eventBroker, state::WorldState &worldState) override;
  void registerGeometryBoundary(std::unique_ptr<Geometry> geometry);
  void resetGeometryBoundary();
  void cancelEvents();

  bool moving() const;
  sro::Position position() const override;
  sro::Position positionAtTime(const PacketContainer::Clock::time_point &timestamp) const;
  float currentSpeed() const;
  sro::Position positionAfterTime(float seconds) const;

  void setSpeed(float walkSpeed, float runSpeed, const PacketContainer::Clock::time_point &timestamp);
  void setAngle(sro::Angle angle);
  void setMotionState(entity::MotionState motionState, const PacketContainer::Clock::time_point &timestamp);
  void setStationaryAtPosition(const sro::Position &position);
  void syncPosition(const sro::Position &position,
                    const PacketContainer::Clock::time_point &timestamp);
  virtual void setMovingToDestination(const std::optional<sro::Position> &sourcePosition,
                                      const sro::Position &destinationPosition,
                                      const PacketContainer::Clock::time_point &timestamp);
  virtual void setMovingTowardAngle(const std::optional<sro::Position> &sourcePosition,
                                    const sro::Angle angle,
                                    const PacketContainer::Clock::time_point &timestamp);
  void movementTimerCompleted();
  void handleEvent(const event::Event *event);
protected:
  bool moving_{false};

  // Only cancels movement timers and sets internal state; does not send any events.
  void privateCancelEvents();
  virtual void cancelMovement();
  sro::Position interpolateCurrentPosition(const PacketContainer::Clock::time_point &currentTime) const;
  float privateCurrentSpeed() const;
  void privateSetStationaryAtPosition(const sro::Position &position);
  void privateSetMovingToDestination(const std::optional<sro::Position> &sourcePosition,
                                     const sro::Position &destinationPosition,
                                     const PacketContainer::Clock::time_point &timestamp);
  void privateSetMovingTowardAngle(const std::optional<sro::Position> &sourcePosition,
                                   const sro::Angle angle,
                                   const PacketContainer::Clock::time_point &timestamp);
private:
  std::optional<broker::EventBroker::SubscriptionId> movementTimerEndedSubscription_;
  std::unique_ptr<Geometry> geometry_;
  std::optional<broker::EventBroker::EventId> enterGeometryEventId_;
  std::optional<broker::EventBroker::EventId> exitGeometryEventId_;
  void checkIfWillCrossGeometryBoundary();
  void cancelGeometryEvents();
};

std::ostream& operator<<(std::ostream &stream, MotionState motionState); // TODO: Replace with toString()

} // namespace entity

#endif // ENTITY_MOBILE_ENTITY_HPP_