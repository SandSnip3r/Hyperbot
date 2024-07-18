#ifndef ENTITY_ENTITY_HPP_
#define ENTITY_ENTITY_HPP_

#include "geometry.hpp"
#include "broker/eventBroker.hpp"
#include "broker/timerManager.hpp"
#include "pk2/characterData.hpp"

#include <silkroad_lib/entity.h>
#include <silkroad_lib/position.h>
#include <silkroad_lib/scalar_types.h>

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

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
  std::optional<broker::EventBroker::EventId> movingEventId;
  void initializeAsMoving(const sro::Position &destinationPosition);
  void initializeAsMoving(sro::Angle destinationAngle);
  void registerGeometryBoundary(std::unique_ptr<Geometry> geometry, broker::EventBroker &eventBroker);
  void resetGeometryBoundary(broker::EventBroker &eventBroker);
  void cancelEvents(broker::EventBroker &eventBroker);

  bool moving() const;
  virtual sro::Position position() const override;
  float currentSpeed() const;
  sro::Position positionAfterTime(float seconds) const;

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

  // Only cancels movement timers and sets internal state; does not send any events.
  void privateCancelEvents(broker::EventBroker &eventBroker);
  void cancelMovement(broker::EventBroker &eventBroker);
  sro::Position interpolateCurrentPosition(const std::chrono::high_resolution_clock::time_point &currentTime) const;
  float privateCurrentSpeed() const;
  void privateSetStationaryAtPosition(const sro::Position &position, broker::EventBroker &eventBroker);
  void privateSetMovingToDestination(const std::optional<sro::Position> &sourcePosition, const sro::Position &destinationPosition, broker::EventBroker &eventBroker);
  void privateSetMovingTowardAngle(const std::optional<sro::Position> &sourcePosition, const sro::Angle angle, broker::EventBroker &eventBroker);
private:
  std::unique_ptr<Geometry> geometry_;
  std::optional<broker::EventBroker::EventId> enterGeometryEventId_;
  std::optional<broker::EventBroker::EventId> exitGeometryEventId_;
  void checkIfWillCrossGeometryBoundary(broker::EventBroker &eventBroker);
  void cancelGeometryEvents(broker::EventBroker &eventBroker);
};

class Character : public MobileEntity {
public:
  sro::entity::LifeState lifeState;
  bool knowCurrentHp() const;
  uint32_t currentHp() const;

  void setLifeState(sro::entity::LifeState newLifeState, broker::EventBroker &eventBroker);
  void setCurrentHp(uint32_t hp, broker::EventBroker &eventBroker);

  // ---- Buffs ----
  struct BuffData {
    sro::scalar_types::ReferenceObjectId skillRefId;
    std::chrono::high_resolution_clock::time_point endTimePoint;
  };
  // Maps TokenId to BuffData
  std::map<uint32_t, BuffData> buffDataMap;
  std::set<sro::scalar_types::ReferenceObjectId> activeBuffs() const;
  bool buffIsActive(sro::scalar_types::ReferenceObjectId skillRefId) const;
  int buffMsRemaining(sro::scalar_types::ReferenceObjectId skillRefId) const;
  void addBuff(sro::scalar_types::ReferenceObjectId skillRefId, uint32_t tokenId, int32_t durationMs, broker::EventBroker &eventBroker);
  void removeBuff(sro::scalar_types::ReferenceObjectId skillRefId, uint32_t tokenId, broker::EventBroker &eventBroker);
  void clearBuffs();
protected:
  std::optional<uint32_t> currentHp_;
};

class PlayerCharacter : public Character {
public:
  std::string name;
};

class NonplayerCharacter : public Character {};

class Monster : public NonplayerCharacter {
public:
  uint32_t getMaxHp(const pk2::CharacterData &characterData) const;
  sro::entity::MonsterRarity rarity;
  std::optional<sro::scalar_types::EntityGlobalId> targetGlobalId;
};

class Item : public Entity {
public:
  sro::entity::ItemRarity rarity;
  std::optional<uint32_t> ownerJId;
  void removeOwnership(broker::EventBroker &eventBroker);
};

class Portal : public Entity {
public:
  uint8_t unkByte3;
};

std::ostream& operator<<(std::ostream &stream, MotionState motionState);


} // namespace entity

#endif // ENTITY_ENTITY_HPP_