#include "castSkill.hpp"
#include "training.hpp"

#include "common/random.hpp"
#include "entity/geometry.hpp"
#include "bot.hpp"
#include "helpers.hpp"
#include "packet/building/clientAgentActionSelectRequest.hpp"
#include "packet/building/clientAgentCharacterMoveRequest.hpp"
#include "packet/building/clientAgentCharacterUpdateBodyStateRequest.hpp"
#include "state/machine/applyStatPoints.hpp"
#include "state/machine/pickItem.hpp"
#include "state/machine/pickItemWithCos.hpp"
#include "type_id/categories.hpp"

// Pathfinder
#include "math_helpers.h"

#include <silkroad_lib/constants.hpp>
#include <silkroad_lib/position_math.hpp>

#include <absl/log/log.h>
#include <absl/strings/str_join.h>

namespace {

sro::Position randomPointInGeometry(const entity::Geometry *geometry) {
  // TODO: This could end up giving us a position outside of the geometry because of the way positions are converted when sent over the network
  const auto *circle = dynamic_cast<const entity::Circle*>(geometry);
  if (circle == nullptr) {
    throw std::runtime_error("Not yet generating random points in non-circle geometries");
  }
  // Pick a random point inside a square that encloses our training circle
  static auto eng = common::createRandomEngine();
  auto notInCircle = [](double x, double y, double radius) {
    return sqrt(x*x+y*y) > radius;
  };
  std::uniform_real_distribution<double> dist(-circle->radius(), circle->radius());
  auto x = dist(eng);
  auto y = dist(eng);
  while (notInCircle(x, y, circle->radius())) {
    x = dist(eng);
    y = dist(eng);
  }
  // Transform x,y to sro coordinate
  return sro::position_math::createNewPositionWith2dOffset(circle->center(), x, y);
}

} // anonymous namespace

namespace state::machine {

void Training::resetSkillLists() {
  skillsToUse_.clear();
  trainingBuffs_.clear();
  nonTrainingBuffs_.clear();
  imbueRefId_.reset();
}

void Training::getSkillsFromConfig() {
  const proto::character_config::CharacterConfig &characterConfig = bot_.config()->proto();
  const proto::character_config::TrainingConfig &trainingConfig = characterConfig.training_config();

  // Attack skills
  for (sro::scalar_types::ReferenceObjectId id : trainingConfig.training_attack_skill_ids()) {
    skillsToUse_.push_back(id);
  }
  removeSkillsFromListWhichWeDontHave(skillsToUse_);

  // Training buffs
  for (sro::scalar_types::ReferenceObjectId id : trainingConfig.training_buff_skill_ids()) {
    const auto &skill = bot_.gameData().skillData().getSkillById(id);
    if (skill.isImbue()) {
      if (imbueRefId_.has_value()) {
        LOG(WARNING) << "Multiple imbues specified in config: " << *imbueRefId_ << " and " << id;
      }
      imbueRefId_ = id;
    } else {
      trainingBuffs_.push_back(id);
    }
  }
  removeSkillsFromListWhichWeDontHave(trainingBuffs_);

  // Nontraining buffs
  for (sro::scalar_types::ReferenceObjectId id : trainingConfig.nontraining_buff_skill_ids()) {
    nonTrainingBuffs_.push_back(id);
  }
  removeSkillsFromListWhichWeDontHave(nonTrainingBuffs_);

  // Move all buffs which require a weapon at all times to the end
  // TODO: Can do better, move skills which require a weapon at all times AND have a cooldown shorter than the skill duration even further to the back of the list
  auto sortBuffs = [&](auto &buffList) {
    int backIndex = buffList.size()-1;
    for (int i=0; i<backIndex; ++i) {
      const auto &buffRefId = buffList[i];
      const auto &buffData = bot_.gameData().skillData().getSkillById(buffRefId);
      const auto buffRequiredWeapons = buffData.reqi();
      if (!buffRequiredWeapons.empty()) {
        // Buff requires a weapon at all times
        LOG(INFO) << "Moving buffs around. Moving " << i << " to " << backIndex;
        std::swap(buffList[i], buffList[backIndex]);
        --backIndex;
      }
    }
  };
  sortBuffs(trainingBuffs_);
  sortBuffs(nonTrainingBuffs_);

  LOG(INFO) << "Skills parsed from config:";
  if (imbueRefId_) {
    LOG(INFO) << "  Imbue: " << *imbueRefId_;
  }
  LOG(INFO) << "  TrainingAttackSkills: " << absl::StrJoin(trainingConfig.training_attack_skill_ids(), ", ");
  LOG(INFO) << "  TrainingBuffSkills: " << absl::StrJoin(trainingConfig.training_buff_skill_ids(), ", ");
  LOG(INFO) << "  NontrainingBuffSkills: " << absl::StrJoin(trainingConfig.nontraining_buff_skill_ids(), ", ");
}

Training::Training(Bot &bot, std::unique_ptr<entity::Geometry> &&trainingAreaGeometry) : StateMachine(bot), trainingAreaGeometry_(std::move(trainingAreaGeometry)) {
  if (bot_.config() == nullptr) {
    throw std::runtime_error("Cannot construct Training state machine if Bot does not have a config");
  }
  stateMachineCreated(kName);
  getSkillsFromConfig();

  bot_.selfState()->setTrainingAreaGeometry(trainingAreaGeometry_->clone());
  bot_.selfState()->registerGeometryBoundary(bot_.selfState()->trainingAreaGeometry->clone());
  // Register training area geometry with all entities
  for (const auto &idPtrPair : bot_.worldState().entityTracker().getEntityMap()) {
    if (idPtrPair.second) {
      if (auto *mobileEntity = dynamic_cast<entity::MobileEntity*>(idPtrPair.second.get())) {
        mobileEntity->registerGeometryBoundary(bot_.selfState()->trainingAreaGeometry->clone());
      }
    }
  }
}

Training::~Training() {
  bot_.selfState()->resetTrainingAreaGeometry();
  bot_.selfState()->resetGeometryBoundary();
  for (const auto &idPtrPair : bot_.worldState().entityTracker().getEntityMap()) {
    if (idPtrPair.second) {
      if (auto *mobileEntity = dynamic_cast<entity::MobileEntity*>(idPtrPair.second.get())) {
        mobileEntity->resetGeometryBoundary();
      }
    }
  }
  stateMachineDestroyed();
}

Status Training::onUpdate(const event::Event *event) {
  // TODO: Improve on this mechanism
  // When we recurse, we do not want to handle the same event again.
  // If we passed nullptr as our event, we would solve this problem, but we do want to pass that event to the presumably newly created child state machine.
  // So, instead, we keep track of the events we've already handled, and only handle the event if we haven't yet handled it.
  // This problem is currently only unique to this state machine because this is the only one which does some event processing before forwarding the event to the child state machine.
  struct RAII {
    RAII(std::set<const event::Event*> &handledEvents, const event::Event *event) : handledEvents_(handledEvents), event_(event) {
      if (event_ != nullptr) {
        if (handledEvents_.find(event_) == handledEvents_.end()) {
          // Not yet handled
          handledEvents_.emplace(event_);
          placed_ = true;
        }
      }
    }
    ~RAII() {
      if (event_ != nullptr && placed_) {
        auto it = handledEvents_.find(event_);
        if (it == handledEvents_.end()) {
          throw std::runtime_error("We thought we placed this event");
        }
        handledEvents_.erase(it);
      }
    }
    std::set<const event::Event*> &handledEvents_;
    const event::Event * const event_;
    bool placed_{false};
  };

  // Check if this event updates our internal state.
  if (event != nullptr && handledEvents_.find(event) == handledEvents_.end()) {
    if (const auto *entitySpawnedEvent = dynamic_cast<const event::EntitySpawned*>(event)) {
      std::shared_ptr<entity::Entity> entity = bot_.worldState().getEntity(entitySpawnedEvent->globalId);
      if (auto *mobileEntity = dynamic_cast<entity::MobileEntity*>(entity.get())) {
        // Register our training region boundary with the entity so that we will get events if they enter/exit
        mobileEntity->registerGeometryBoundary(bot_.selfState()->trainingAreaGeometry->clone());
      }
    } else if (const auto *entityMovementBegan = dynamic_cast<const event::EntityMovementBegan*>(event)) {
      if (walkingTargetAndAttack_ && entityMovementBegan->globalId == walkingTargetAndAttack_->targetId) {
        // The target that we're walking to has changed its movement, we'll need to recalculate
        walkingTargetAndAttack_.reset();
      }
    } else if (const auto *entityLifeStateChanged = dynamic_cast<const event::EntityLifeStateChanged*>(event)) {
      if (entityLifeStateChanged->globalId == bot_.selfState()->globalId) {
        if (bot_.selfState()->lifeState == sro::entity::LifeState::kDead) {
          // We died. Cleanup and gtfo.
          return Status::kDone;
        }
      }
    // } else if (const auto *configUpdated = dynamic_cast<const event::ConfigUpdated*>(event)) {
    //   LOG(INFO) << "Config updated, resetting skills and re-fetching from config";
    //   resetSkillLists();
    //   getSkillsFromConfig();
    } else if (event->eventCode == event::EventCode::kCharacterAvailableStatPointsUpdated) {
      if (!applyStatPointsChildStateMachine_ && bot_.selfState()->getAvailableStatPoints() > 0) {
        LOG(INFO) << "Constructing state machine to apply " << bot_.selfState()->getAvailableStatPoints() << " stat points to int";
        applyStatPointsChildStateMachine_ = std::make_unique<ApplyStatPoints>(bot_, std::vector<StatPointType>(bot_.selfState()->getAvailableStatPoints(), StatPointType::kInt));
      }
    }
  }
  // Mark this event as handled, so that if we recurse, we dont handle it again
  RAII raii(handledEvents_, event);

  if (applyStatPointsChildStateMachine_) {
    const Status status = applyStatPointsChildStateMachine_->onUpdate(event);
    if (status == Status::kDone) {
      LOG(INFO) << "Done applying stat points";
      applyStatPointsChildStateMachine_.reset();
    }
  }

  if (childState_) {
    // Either casting a skill, picking up an item ourself, picking up an item with COS, or walking somewhere.
    const Status status = childState_->onUpdate(event);
    if (status == Status::kDone) {
      childState_.reset();
    } else {
      // Child state is not yet done.
      if (dynamic_cast<Walking*>(childState_.get()) == nullptr) {
        // The Walking state machine is always interruptable. For every other state machine, we have nothing else to do in this function.
        return Status::kNotDone;
      }
    }
  }

  // We either have no child state, or we're walking somewhere.
  if (!bot_.selfState()->spawned()) {
    // We are not spawned, nothing to do
    return Status::kNotDone;
  }

  if (bot_.selfState()->lifeState == sro::entity::LifeState::kDead) {
    // Dead, nothing to do.
    // TODO: Maybe go to town after some time
    return Status::kNotDone;
  }

  if (bot_.needToGoToTown()) {
    return Status::kDone;
  }

  // First, check that our buffs are all active. Use different sets of buffs for inside and outside of the training area.
  SkillList *buffList;
  if (trainingAreaGeometry_->pointIsInside(bot_.selfState()->position())) {
    buffList = &trainingBuffs_;
  } else {
    buffList = &nonTrainingBuffs_;
  }

  bool setNewChildState = checkBuffs(*buffList);
  if (setNewChildState) {
    // Have some buff to cast
    return onUpdate(event);
  }

  // TODO: Maybe we should have an event for when we enter the training area. That would trigger training area buffs immediately once we enter.
  // Buffs are all good at this point. First, check if we need to walk to the training area.
  if (!trainingAreaGeometry_->pointIsInside(bot_.selfState()->position())) {
    // We are not in the training area.
    // Are we already walking there?
    if (childState_ && dynamic_cast<Walking*>(childState_.get()) != nullptr) {
      // We are already walking there. Nothing else to do.
      // TODO: We assume that this Walking is to the training area.
      return Status::kNotDone;
    } else if (childState_) {
      throw std::runtime_error("We're not in the training area, not walking to it, all buffs are good, but we still have some child state?");
    }

    // Walk to the training area
    const auto *trainingAreaCircle = dynamic_cast<const entity::Circle*>(trainingAreaGeometry_.get());
    if (trainingAreaCircle == nullptr) {
      throw std::runtime_error("Not sure where to navigate to for other shapes");
    }
    // Not in training area, navigating to the center of the training area
    const auto pathToTrainingAreaCenter = bot_.calculatePathToDestination(trainingAreaCircle->center());
    setChildStateMachine<Walking>(pathToTrainingAreaCenter);
    return onUpdate(event);
  }

  // We're in the training area.
  const auto [itemsInRange, monstersInRange] = getItemsAndMonstersInRange();

  setNewChildState = tryPickItem(itemsInRange);
  if (setNewChildState) {
    return onUpdate(event);
  } else if (walkingToItemTarget_) {
    // We didn't just create a child state machine, but we are already walking to pick up an item.
    return Status::kNotDone;
  }

  setNewChildState = tryAttackMonster(monstersInRange);
  if (setNewChildState) {
    return onUpdate(event);
  } else if (walkingTargetAndAttack_) {
    // We didn't just create a child state machine, but we are already walking to attack a target.
    return Status::kNotDone;
  }

  // TODO: Calculate if entire training geometry is visible, if so, do not explore.
  // Nothing left to do except explore the training area. Randomly walk to a point.
  if (childState_ && dynamic_cast<Walking*>(childState_.get()) != nullptr) {
    // Already walking somewhere.
    return Status::kNotDone;
  }

  setNewChildState = walkToRandomPoint();
  if (setNewChildState) {
    return onUpdate(event);
  }
  LOG(INFO) << "Made it to the end";
  return Status::kNotDone;
}

bool Training::tryPickItem(const ItemList &itemList) {
  if (itemList.empty()) {
    // Nothing in range to pick up.
    return false;
  }

  // Maybe going to pick up an item, check if we're even in a state to be picking up items
  std::optional<sro::scalar_types::EntityGlobalId> cosGlobalId;
  if (!bot_.selfState()->cosInventoryMap.empty()) {
    // Have a cos
    // TODO: How do we pick which COS to use?
    //  Maybe there can only ever be one.
    //  For now, just use the "first"
    cosGlobalId = bot_.selfState()->cosInventoryMap.begin()->first;
  }
  if (!cosGlobalId && !canMove()) {
    // No way to pick up items. No pickpet and cant move.
    return false;
  }

  // Have some option to pick item
  // Lets see if there's anything to pick up
  auto wantItem = [](const entity::Item *item) {
    // TODO: Check if this item should be picked up, based on a configured filter
    return true;
    // return (item->refObjId != 62 && // Arrows
    //         item->refObjId != 10383); // Bolts
  };
  for (const auto *item : itemList) {
    if (!item->ownerJId || *item->ownerJId == bot_.selfState()->jId) {
      // No owner or it belongs to us. Pick it up
      // TODO: Track party members' JIDs so that we know if we can pick up their items.
      if (!wantItem(item)) {
        // Skipping item
        continue;
      }
      const auto targetItemGlobalId = item->globalId;
      if (cosGlobalId) {
        possiblyOverwriteChildStateMachine<PickItemWithCos>(*cosGlobalId, targetItemGlobalId);
      } else {
        // We must pick the item ourself.
        // First, check if we're already walking to this item.
        bool justFinishedWalking{false};
        if (walkingToItemTarget_ && *walkingToItemTarget_ == targetItemGlobalId) {
          // We were/are walking to pick an item.
          if (childState_) {
            // Still walking to it.
            return false;
          } else {
            // We just finished walking to it.
            justFinishedWalking = true;
            walkingToItemTarget_.reset();
          }
        }
        // We are not walking to this item. We need to either start walking to it or pick it up.
        std::shared_ptr<entity::Item> item = bot_.worldState().getEntity<entity::Item>(targetItemGlobalId);
        const auto itemPosition = item->position();
        const auto distanceToItem = sro::position_math::calculateDistance2d(bot_.selfState()->position(), itemPosition);
        const float kMinimumDistanceToPickItem{5.0}; // TODO: Figure out more precisely what works here. Ideally this distance should be as large as possible without going over.
        if (justFinishedWalking || distanceToItem <= kMinimumDistanceToPickItem) {
          // Just got to the item, or within range of it; pick it up.
          possiblyOverwriteChildStateMachine<PickItem>(targetItemGlobalId);
        } else {
          // Need to walk to the item.
          const auto pathToItem = bot_.calculatePathToDestination(itemPosition);
          possiblyOverwriteChildStateMachine<Walking>(pathToItem);
          walkingToItemTarget_ = targetItemGlobalId;
        }
      }
      return true;
    }
  }

  return false;
}

bool Training::tryAttackMonster(const MonsterList &monsterList) {
  if (monsterList.empty()) {
    // No monsters in range to attack
    return false;
  }

  // Evaluate targets and skills
  const auto targetAndAttack = getTargetAndAttackSkill(monsterList);
  if (!targetAndAttack) {
    // No target/skill chosen
    return false;
  }
  bool finishedWalking{false};
  if (walkingTargetAndAttack_ && walkingTargetAndAttack_->skillId == targetAndAttack->skillId && walkingTargetAndAttack_->targetId == targetAndAttack->targetId) {
    if (childState_) {
      // We're already walking to this goal.
      return false;
    } else {
      // We finished walking to this goal
      finishedWalking = true;
      walkingTargetAndAttack_.reset();
    }
  }
  std::shared_ptr<const entity::MobileEntity> targetEntity = bot_.worldState().getEntity<entity::MobileEntity>(targetAndAttack->targetId);
  const auto attackRefId = targetAndAttack->skillId;
  std::optional<sro::Position> destinationPosition;
  if (!finishedWalking) {
    destinationPosition = calculateWhereToWalkToAttackEntityWithSkill(*targetEntity.get(), attackRefId);
  }

  if (destinationPosition) {
    // Need to walk to cast skill on entity.
    // LOG(INFO) << "Need to walk to " << *destinationPosition << " to cast skill on entity, which is " << sro::position_math::calculateDistance2d(targetEntity.position(), *destinationPosition) << " away from the target";
    const auto pathToDestination = bot_.calculatePathToDestination(*destinationPosition);
    possiblyOverwriteChildStateMachine<Walking>(pathToDestination);
    walkingTargetAndAttack_ = targetAndAttack;
  } else {
    // We are within range to cast skill.
    if (bot_.selfState()->hwanPoints() == 5) {
      // TODO: Use berserk in a StateMachine.
      const auto packet = packet::building::ClientAgentCharacterUpdateBodyStateRequest::packet(packet::enums::BodyState::kHwan);
      bot_.proxy().inject(packet, PacketContainer::Direction::kBotToServer);
    }
    CastSkillStateMachineBuilder castSkillBuilder(bot_, attackRefId);
    castSkillBuilder.withTarget(targetEntity->globalId);
    const auto &skillData = bot_.gameData().skillData().getSkillById(attackRefId);
    const auto weaponSlot = getInventorySlotOfWeaponForSkill(skillData, bot_);
    if (weaponSlot) {
      castSkillBuilder.withWeapon(*weaponSlot);

      // Check if the weapon can be accompanied by a shield
      const auto *item = bot_.selfState()->inventory.getItem(*weaponSlot);
      const storage::ItemEquipment *equipment = dynamic_cast<const storage::ItemEquipment*>(item);
      if (equipment != nullptr) {
        if (!equipment->itemInfo->twoHanded) {
          const auto shieldSlot = getInventorySlotOfShield(bot_);
          if (shieldSlot) {
            // Inventory slot of shield for skill is `*shieldSlot`
            castSkillBuilder.withShield(*shieldSlot);
          }
        }
      }
    }

    // TODO: Maybe the attack requires a shield.

    if (imbueRefId_) {
      castSkillBuilder.withImbue(*imbueRefId_);
    }
    if (targetEntity->globalId != bot_.selfState()->globalId && (!bot_.selfState()->selectedEntity || *bot_.selfState()->selectedEntity != targetEntity->globalId)) {
      const auto selectPacket = packet::building::ClientAgentActionSelectRequest::packet(targetEntity->globalId);
      bot_.packetBroker().injectPacket(selectPacket, PacketContainer::Direction::kBotToServer);
    }
    possiblyOverwriteChildStateMachine(castSkillBuilder.create());
  }
  return true;
}

bool Training::walkToRandomPoint() {
  if (!canMove()) {
    return false;
  }

  constexpr const int kMaxTryCount{10};
  bool success{false};
  for (int i=0; i<kMaxTryCount; ++i) {
    try {
      const auto pathToRandomPoint = bot_.calculatePathToDestination(randomPointInGeometry(trainingAreaGeometry_.get()));
      setChildStateMachine<Walking>(pathToRandomPoint);
      success = true;
      break;
    } catch (std::exception &ex) {
      LOG(ERROR) << "Couldn't walk to random position. Exception: " << ex.what();
    }
  }
  if (!success) {
    LOG(INFO) << "Tried " << kMaxTryCount << " times, cannot walk to a random point";
  }
  return success;
}

std::tuple<Training::ItemList, Training::MonsterList> Training::getItemsAndMonstersInRange() const {
  ItemList itemsInRange;
  MonsterList monstersInRange;
  for (const auto &entityIdPtrPair : bot_.worldState().entityTracker().getEntityMap()) {
    if (const auto *monster = dynamic_cast<const entity::Monster*>(entityIdPtrPair.second.get())) {
      if (monster->lifeState != sro::entity::LifeState::kAlive) {
        // Dont care about monsters that aren't alive
        continue;
      }
      if (monster->currentHpIsKnown() && monster->currentHp() == 0) {
        // Monster isn't dead, but has 0 hp. It's effectively dead to us.
        continue;
      }
      if (wantToAttackMonster(*monster)) {
        monstersInRange.push_back(monster);
      }
    } else if (const auto *item = dynamic_cast<const entity::Item*>(entityIdPtrPair.second.get())) {
      if (trainingAreaGeometry_->pointIsInside(item->position())) {
        itemsInRange.push_back(item);
      }
    }
  }
  return {itemsInRange, monstersInRange};
}

// Returns true if it set a new child state.
bool Training::checkBuffs(const SkillList &buffList) {
  const auto nextBuffToCast = getNextBuffToCast(buffList);
  if (nextBuffToCast) {
    auto castSkillBuilder = CastSkillStateMachineBuilder(bot_, *nextBuffToCast);
    // Next buff to cast is `*nextBuffToCast`
    const auto &buffData = bot_.gameData().skillData().getSkillById(*nextBuffToCast);

    // Does the buff require a specific weapon to be equipped to cast?
    const auto weaponSlot = getInventorySlotOfWeaponForSkill(buffData, bot_);
    // Note: It is also possible that a skill requires a shield (shield bash)
    if (weaponSlot) {
      // Inventory slot of weapon for skill is `*weaponSlot`
      castSkillBuilder.withWeapon(*weaponSlot);

      // Check if the weapon can be accompanied by a shield
      const auto *item = bot_.selfState()->inventory.getItem(*weaponSlot);
      const storage::ItemEquipment *equipment = dynamic_cast<const storage::ItemEquipment*>(item);
      if (equipment != nullptr) {
        if (!equipment->itemInfo->twoHanded) {
          const auto shieldSlot = getInventorySlotOfShield(bot_);
          if (shieldSlot) {
            // Inventory slot of shield for skill is `*shieldSlot`
            castSkillBuilder.withShield(*shieldSlot);
          }
        }
      } else {
        LOG(WARNING) << "Found item (supposed to be weapon) at slot " << static_cast<int>(*weaponSlot) << " is not a piece of equipment";
      }
    }

    // TODO: Does the skill actually require a shield? If so, equip it.

    if (buffData.targetRequired) {
      // TODO: We assume this buff is for ourself
      // Buff requires a target, using self
      castSkillBuilder.withTarget(bot_.selfState()->globalId);
    }

    // Create a child state to cast skill
    possiblyOverwriteChildStateMachine(castSkillBuilder.create());
    return true;
  }

  // Nothing to do.
  return false;
}

void Training::possiblyOverwriteChildStateMachine(std::unique_ptr<StateMachine> newChildStateMachine) {
  if (childState_ && dynamic_cast<Walking*>(childState_.get()) == nullptr) {
    throw std::runtime_error("Cannot overwrite a child state which is not Walking");
  }
  walkingTargetAndAttack_.reset();
  setChildStateMachine(std::move(newChildStateMachine));
}

std::optional<sro::Position> Training::calculateWhereToWalkToAttackEntityWithSkill(const entity::MobileEntity &entity, sro::scalar_types::ReferenceObjectId attackRefId) {
  const auto selfCurrentPosition = bot_.selfState()->position();
  const auto &skill = bot_.gameData().skillData().getSkillById(attackRefId);
  const double skillRangeMinusRounding = std::max(sro::constants::kSqrtHalf, skill.actionRange - sro::constants::kSqrtHalf*2);
  const auto calcd_dist = sro::position_math::calculateDistance2d(selfCurrentPosition, entity.position());
  if (calcd_dist <= skillRangeMinusRounding) {
    // We are already close enough.
    return {};
  }

  auto calculateDestination = [&](const sro::Position &targetPos) {
    // We don't need to walk completely to the position of the target, instead, we will take the path to the target, and find the earliest point on that path which is within range of the target.
    const auto pathToDestination = bot_.calculatePathToDestination(targetPos);
    if (pathToDestination.size() < 2) {
      throw std::runtime_error("Path to target has fewer than two points. We expect this path to at least contain the starting pos and end pos.");
    }

    if (skill.actionRange == 0.0) {
      // Need to walk to the exact position of the target.
      return packet::building::NetworkReadyPosition::roundToNearest(targetPos).asSroPosition();
    }

    // Find the farthest point, within range, on the last straight segment towards the target.
    const auto index = pathToDestination.size()-2;
    const auto indexSroPos = pathToDestination.at(index).asSroPosition();
    if (sro::position_math::calculateDistance2d(targetPos, indexSroPos) <= skillRangeMinusRounding) {
      // This position is within range of the target, use this.
      // TODO: Due to the way we break up long movements, I should iterate and see if the previous line segment is in line with this one, or the one before that.
      //  Added complication, the calculated path has already rounded waypoints to integer coordinates, so it might be hard to check if two segments are actually aligned.
      return indexSroPos;
    }

    // Find where the line pathToDestination[index] -> pathToDestination[index+1] intersects with the circle defined around the target with the radius as the range of the skill.
    // First, convert both points to pathfinder::Vectors.
    const auto [res0X, res0Z] = sro::position_math::calculateOffsetInOtherRegion(indexSroPos, targetPos);
    const auto [res1X, res1Z] = sro::position_math::calculateOffsetInOtherRegion(pathToDestination.at(index+1).asSroPosition(), targetPos);
    const pathfinder::Vector res0Vector(res0X, res0Z), res1Vector(res1X, res1Z), centerOfCircleVector(targetPos.xOffset(), targetPos.zOffset());
    pathfinder::Vector out0, out1;
    const int intersectionCount = pathfinder::math::lineSegmentIntersectsWithCircle(res1Vector, res0Vector, centerOfCircleVector, skillRangeMinusRounding, &out0, &out1);
    if (intersectionCount != 1) {
      throw std::runtime_error("Expecting there to be exactly one intersection point");
    }
    auto destinationSroPos = sro::Position(targetPos.regionId(), out0.x(), targetPos.yOffset(), out0.y()); // TODO: Y is bogus.
    return packet::building::NetworkReadyPosition::truncateForNetwork(destinationSroPos);
  };

  auto destinationPos = calculateDestination(entity.position());
  if (entity.moving()) {
    // Run an iterative algorithm to walk to where the target will be by the time we get there.
    int remainingRecalculationCount = 10;
    const float kDistanceTolerance = 0.01;
    while (1) {
      // We already calculated a destination that we want to walk to.
      // First calculate how long it will take us to get there.
      const auto distanceToDestination = sro::position_math::calculateDistance2d(selfCurrentPosition, destinationPos);
      constexpr const float kEstimatedPingTimeSeconds = 0.080;
      const auto timeToGetToCurrentDest = distanceToDestination / bot_.selfState()->currentSpeed() + kEstimatedPingTimeSeconds;

      // Next, calculate where the target will be after that amount of time.
      const auto targetNewPos = entity.positionAfterTime(timeToGetToCurrentDest);

      // Now, check if our calculated destination is close enough to the predicted target position.
      const auto distanceToNewTargetPos = sro::position_math::calculateDistance2d(destinationPos, targetNewPos);
      if (std::abs(skillRangeMinusRounding-distanceToNewTargetPos) <= kDistanceTolerance) {
        // We want to get as close as possible to the shortest distance.
        //  If the target is running away from us, we'll calculate that we need to go farther.
        //  If the target is running at us, we'll calculate that we dont need to go as far.
        // We're close enough
        break;
      }

      // Not close enough, recalculate and try to go to where the target will be
      destinationPos = calculateDestination(targetNewPos);

      // Limit the number of iterations of this algorithm.
      --remainingRecalculationCount;
      if (remainingRecalculationCount == 0) {
        // Exhausted allowed computation limit
        break;
      }
    }
  }
  return destinationPos;
}

bool Training::wantToAttackMonster(const entity::Monster &monster) const {
  const auto &characterData = bot_.gameData().characterData().getCharacterById(monster.refObjId);

  // TODO: Move this to the config, or determine at runtime.
  // if (std::abs(characterData.lvl - bot_.selfState()->getCurrentLevel()) > 10) {
  //   // Don't attack anything 10 levels above or below us
  //   return false;
  // }

  return trainingAreaGeometry_->pointIsInside(monster.position());
}

void Training::removeSkillsFromListWhichWeDontHave(SkillList &skillList) {
  auto newEndIt = std::remove_if(skillList.begin(), skillList.end(), [&](const auto &skillId) {
    return !bot_.selfState()->haveSkill(skillId);
  });
  for (auto it=newEndIt; it!=skillList.end(); ++it) {
    LOG(INFO) << "Dont have skill " << bot_.gameData().getSkillName(*it) << ". Removing from list";
  }
  skillList.erase(newEndIt, skillList.end());
};

std::optional<sro::scalar_types::ReferenceObjectId> Training::getNextBuffToCast(const SkillList &buffList) const {
  // Lets evaluate our buffs and see if any need to be reactivated
  auto copyOfBuffsToUse = buffList;
  // Remove all buffs which are currently active.
  copyOfBuffsToUse.erase(std::remove_if(copyOfBuffsToUse.begin(), copyOfBuffsToUse.end(), [&](const auto &buff) {
    return bot_.selfState()->buffIsActive(buff);
  }), copyOfBuffsToUse.end());

  if (copyOfBuffsToUse.empty()) {
    // No buffs need to be cast
    return {};
  }

  // Choose one that isnt active, on cooldown, or already used
  for (int i=0; i<copyOfBuffsToUse.size(); ++i) {
    const auto buff = copyOfBuffsToUse.at(i);
    if (bot_.selfState()->skillEngine.alreadyTriedToCastSkill(buff)) {
      continue;
    }
    if (bot_.similarSkillIsAlreadyActive(buff)) {
      continue;
    }
    if (!bot_.canCastSkill(buff)) {
      // Cannot cast this skill right now. Maybe we're stunned, or it's on cooldown.
      continue;
    }
    return buff;
  }
  // Didn't find a buff that we need to cast.
  return {};
}

// TODO: Also return the path to walk to cast the skill on the target.
std::optional<Training::TargetAndAttackSkill> Training::getTargetAndAttackSkill(const MonsterList &monsters) const {
  if (monsters.empty()) {
    throw std::runtime_error("monsters empty");
  }

  // Figure out which skills we have available to us
  SkillList availableSkills;
  for (const auto skillId : skillsToUse_) {
    if (bot_.canCastSkill(skillId)) {
      availableSkills.push_back(skillId);
    }
  }
  if (availableSkills.empty()) {
    // Want to attack a monster but have no available skills
    return {};
  }

  struct RarityAndAttacking {
    // RarityAndAttacking(sro::entity::MonsterRarity r, bool a) : rarity(r), isAttacking(a) {}
    sro::entity::MonsterRarity rarity;
    bool isAttacking;
  };

  // Lower number is higher priority
  auto getPriorityBasedOnRarity = [this](const entity::Monster *monster) {
    // Earlier in the list is a higher priority
    // 1. Prefer the smallest thing attacking us
    // 2. Prefer the largest thing if nothing attacking us
    static const std::vector<RarityAndAttacking> monsterPriorities {
      {sro::entity::MonsterRarity::kGeneral, true},
      {sro::entity::MonsterRarity::kChampion, true},
      {sro::entity::MonsterRarity::kElite, true},
      {sro::entity::MonsterRarity::kGeneralParty, true},
      {sro::entity::MonsterRarity::kGiant, true},
      {sro::entity::MonsterRarity::kTitan, true},
      {sro::entity::MonsterRarity::kEliteStrong, true},
      {sro::entity::MonsterRarity::kChampionParty, true},
      {sro::entity::MonsterRarity::kGiantParty, true},
      {sro::entity::MonsterRarity::kTitanParty, true},
      {sro::entity::MonsterRarity::kEliteParty, true},
      {sro::entity::MonsterRarity::kUnique, true},
      {sro::entity::MonsterRarity::kUnique2, true},
      {sro::entity::MonsterRarity::kUniqueParty, true},
      {sro::entity::MonsterRarity::kUnique2Party, true},

      {sro::entity::MonsterRarity::kUnique2Party, false},
      {sro::entity::MonsterRarity::kUniqueParty, false},
      {sro::entity::MonsterRarity::kUnique2, false},
      {sro::entity::MonsterRarity::kUnique, false},
      {sro::entity::MonsterRarity::kGiantParty, false},
      {sro::entity::MonsterRarity::kEliteParty, false},
      {sro::entity::MonsterRarity::kTitanParty, false},
      {sro::entity::MonsterRarity::kChampionParty, false},
      {sro::entity::MonsterRarity::kGeneralParty, false},
      {sro::entity::MonsterRarity::kEliteStrong, false},
      {sro::entity::MonsterRarity::kElite, false},
      {sro::entity::MonsterRarity::kTitan, false},
      {sro::entity::MonsterRarity::kGiant, false},
      {sro::entity::MonsterRarity::kChampion, false},
      {sro::entity::MonsterRarity::kGeneral, false},
    };
    const bool isAttackingUs = (monster->targetGlobalId && *monster->targetGlobalId == bot_.selfState()->globalId);
    int index=0;
    for (const auto &i : monsterPriorities) {
      if (i.rarity == monster->rarity && i.isAttacking == isAttackingUs) {
        return index;
      }
      ++index;
    }
    LOG(WARNING) << "Dont know prioritiy of this monster!";
    return 100;
  };

  // Choose the highest priority monster (lowest priority value, tiebreak with which is closer)
  const auto currentPosition = bot_.selfState()->position();
  const auto currentLevel = bot_.selfState()->getCurrentLevel();
  auto tmpMonsterList = monsters;
  std::sort(tmpMonsterList.begin(), tmpMonsterList.end(), [&](const entity::Monster *lhs, const entity::Monster *rhs) {
    // Prefer monsters which are currently attacking us.
    const bool lhsIsAttackingUs = lhs->targetGlobalId && *lhs->targetGlobalId == bot_.selfState()->globalId;
    const bool rhsIsAttackingUs = rhs->targetGlobalId && *rhs->targetGlobalId == bot_.selfState()->globalId;
    if (lhsIsAttackingUs == rhsIsAttackingUs) {
      // Prefer monsters closest to our level.
      const auto lhsLevel = bot_.gameData().characterData().getCharacterById(lhs->refObjId).lvl;
      const auto rhsLevel = bot_.gameData().characterData().getCharacterById(rhs->refObjId).lvl;
      const auto levelDiffLhs = std::abs(currentLevel - lhsLevel);
      const auto levelDiffRhs = std::abs(currentLevel - rhsLevel);
      if (levelDiffLhs == levelDiffRhs) {
        // Prefer monsters with higher priority.
        const auto lhsPriority = getPriorityBasedOnRarity(lhs);
        const auto rhsPriority = getPriorityBasedOnRarity(rhs);
        if (lhsPriority == rhsPriority) {
          // Prefer closer monsters.
          return sro::position_math::calculateDistance2d(currentPosition, lhs->position()) < sro::position_math::calculateDistance2d(currentPosition, rhs->position());
        } else {
          return lhsPriority < rhsPriority;
        }
      } else {
        return levelDiffLhs < levelDiffRhs;
      }
    } else {
      return lhsIsAttackingUs;
    }
  });
  for (const entity::Monster *targetMonster : tmpMonsterList) {
    // Choose the first attack which we can use
    // If the skill applies a buff to the monster (like a DOT), make sure the monster doesn't already have the buff.
    for (const auto skillId : availableSkills) {
      bool canCast = true;
      for (const auto buffId : targetMonster->activeBuffs()) {
        if (buffId == skillId) {
          canCast = false;
          break;
        }
      }
      if (canCast) {
        return TargetAndAttackSkill{targetMonster->globalId, skillId};
      }
    }

  }
  LOG(INFO) << "No available skill/target";
  return {};
}

} // namespace state::machine