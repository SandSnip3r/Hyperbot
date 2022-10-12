#ifndef ENTITY_ENTITY_HPP_
#define ENTITY_ENTITY_HPP_

#include "broker/eventBroker.hpp"
#include "broker/timerManager.hpp"

#include <silkroad_lib/entity.h>
#include <silkroad_lib/position.h>
#include <silkroad_lib/scalar_types.h>

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

namespace entity {

enum class EntityType {
  kSelf,
  kCharacter,
  kPlayerCharacter,
  kNonplayerCharacter,
  kMonster,
  kItem,
  kPortal
};

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

class Entity {
public:
  sro::scalar_types::ReferenceObjectId refObjId;
  uint8_t typeId1, typeId2, typeId3, typeId4;
  sro::scalar_types::EntityGlobalId globalId;
  void initializePosition(const sro::Position &position);
  void initializeAngle(sro::Angle angle);
  virtual sro::Position position() const;
  sro::Angle angle() const;
  virtual ~Entity() = default;
  virtual EntityType entityType() const;
protected:
  sro::Position position_;
  sro::Angle angle_;
};

class MobileEntity : public Entity {
public:
  std::chrono::high_resolution_clock::time_point startedMovingTime;
  std::optional<sro::Position> destinationPosition;
  MotionState motionState;
  std::optional<MotionState> lastMotionState;
  float walkSpeed;
  float runSpeed;
  std::optional<broker::TimerManager::TimerId> movingEventId;
  void initializeAsMoving(const sro::Position &destinationPosition);
  void initializeAsMoving(sro::Angle destinationAngle);

  bool moving() const;
  virtual sro::Position position() const override;
  float currentSpeed() const;

  void setSpeed(float walkSpeed, float runSpeed, broker::EventBroker &eventBroker);
  void setAngle(sro::Angle angle, broker::EventBroker &eventBroker);
  void setMotionState(entity::MotionState motionState, broker::EventBroker &eventBroker);
  void setStationaryAtPosition(const sro::Position &position, broker::EventBroker &eventBroker);
  void syncPosition(const sro::Position &position, broker::EventBroker &eventBroker);
  virtual void setMovingToDestination(const std::optional<sro::Position> &sourcePosition, const sro::Position &destinationPosition, broker::EventBroker &eventBroker);
  virtual void setMovingTowardAngle(const std::optional<sro::Position> &sourcePosition, const sro::Angle angle, broker::EventBroker &eventBroker);
  void movementTimerCompleted(broker::EventBroker &eventBroker);
protected:
  mutable std::mutex mutex_;
  bool moving_;
  void cancelMovement(broker::EventBroker &eventBroker);
  sro::Position interpolateCurrentPosition(const std::chrono::high_resolution_clock::time_point &currentTime) const;
  float privateCurrentSpeed() const;
  void privateSetStationaryAtPosition(const sro::Position &position, broker::EventBroker &eventBroker);
  void privateSetMovingToDestination(const std::optional<sro::Position> &sourcePosition, const sro::Position &destinationPosition, broker::EventBroker &eventBroker);
  void privateSetMovingTowardAngle(const std::optional<sro::Position> &sourcePosition, const sro::Angle angle, broker::EventBroker &eventBroker);
};

class Character : public MobileEntity {
public:
  sro::entity::LifeState lifeState;
  void setLifeState(sro::entity::LifeState newLifeState, broker::EventBroker &eventBroker);
};

class PlayerCharacter : public Character {
public:
  std::string name;
};

class NonplayerCharacter : public Character {};

class Monster : public NonplayerCharacter {
public:
  sro::entity::MonsterRarity rarity;
};

class Item : public Entity {
public:
  sro::entity::ItemRarity rarity;
};

class Portal : public Entity {
public:
  uint8_t unkByte3;
};

std::ostream& operator<<(std::ostream &stream, MotionState motionState);


} // namespace entity

#endif // ENTITY_ENTITY_HPP_