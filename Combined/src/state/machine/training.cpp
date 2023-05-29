#include "castSkill.hpp"
#include "training.hpp"

#include "entity/geometry.hpp"
#include "bot.hpp"
#include "helpers.hpp"
#include "logging.hpp"
#include "packet/building/clientAgentCharacterMoveRequest.hpp"
#include "state/machine/pickItem.hpp"
#include "state/machine/pickItemWithCos.hpp"
#include "type_id/categories.hpp"

// Pathfinder
#include "math_helpers.h"

#include <silkroad_lib/constants.h>
#include <silkroad_lib/position_math.h>

#include <algorithm>
#include <array>
#include <functional>
#include <random>

namespace {

std::mt19937 createRandomEngine() {
  std::random_device rd;
  std::array<int, std::mt19937::state_size> seed_data;
  std::generate_n(seed_data.data(), seed_data.size(), std::ref(rd));
  std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
  return std::mt19937(seq);
}

sro::Position randomPointInGeometry(const entity::Geometry *geometry) {
  // TODO: This could end up giving us a position outside of the geometry because of the way positions are converted when sent over the network
  const auto *circle = dynamic_cast<const entity::Circle*>(geometry);
  if (circle == nullptr) {
    throw std::runtime_error("Not yet generating random points in non-circle geometries");
  }
  // Pick a random point inside a square that encloses our training circle
  static auto eng = createRandomEngine();
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

Training::Training(Bot &bot, std::unique_ptr<entity::Geometry> &&trainingAreaGeometry) : StateMachine(bot), trainingAreaGeometry_(std::move(trainingAreaGeometry)) {
  stateMachineCreated(kName);
  buildBuffList();
  // Create a list of skills to use
  skillsToUse_ = SkillList {
    // TODO: Why do i get a Handle Read Error when I send an invalid skill?
      7805, // Flying Dragon - Flash
      8204, // Crane's Thunderbolt
      8195, // Horse's Thunderbolt
      7675, // Ghost Spear - Emperor
      7672, // Ghost Spear - Storm Cloud
      // 885, // Chain Spear - Shura

      // Rogue
      // 9612, // Distance Shot
      // 9940, // Prick
      // // 9798, // Mortal Wounds
      // // 9623, // Blast Shot
      // // 9631, // Hurricane Shot
      // 9544, // Intense Shot
      // 9868, // Screw
      // 9528, // Power Shot
      // 9587, // Rapid Shot

      // Wizard
      // 10264, // Fire Bolt
      // 10122, // Ice Bolt (very low level)
      // 10135, // Frozen Spear

      // Warrior
      // 8499, // Sprint Assault

      // Bard
      // 11261, // Weird Chord

      // Warlock
      // 11072, // Vampire Kiss
    };

  bot_.selfState().setTrainingAreaGeometry(trainingAreaGeometry_->clone());
  bot_.selfState().registerGeometryBoundary(bot_.selfState().trainingAreaGeometry->clone(), bot_.eventBroker());
  // Register training area geometry with all entities
  for (const auto &idPtrPair : bot_.entityTracker().getEntityMap()) {
    if (idPtrPair.second) {
      if (auto *mobileEntity = dynamic_cast<entity::MobileEntity*>(idPtrPair.second.get())) {
        mobileEntity->registerGeometryBoundary(bot_.selfState().trainingAreaGeometry->clone(), bot_.eventBroker());
      }
    }
  }
}

Training::~Training() {
  bot_.selfState().resetTrainingAreaGeometry();
  bot_.selfState().resetGeometryBoundary(bot_.eventBroker());
  for (const auto &idPtrPair : bot_.entityTracker().getEntityMap()) {
    if (idPtrPair.second) {
      if (auto *mobileEntity = dynamic_cast<entity::MobileEntity*>(idPtrPair.second.get())) {
        mobileEntity->resetGeometryBoundary(bot_.eventBroker());
      }
    }
  }
  stateMachineDestroyed();
}

void Training::onUpdate(const event::Event *event) {
  if (done()) {
    LOG() << "Training's onUpdate called, but it's done" << std::endl;
    return;
  }
  // TODO: Improve on this mechanism
  // When we recurse, we do not want to handle the same event again.
  // If we passed nullptr as our event, we would solve this problem, but we do want to pass that event to the presumably newly created child state machine.
  // So, instead, we keep track of the events we've already handled, and only handle the event if we havent yet handled it.
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
      auto *entity = bot_.entityTracker().getEntity(entitySpawnedEvent->globalId);
      if (auto *mobileEntity = dynamic_cast<entity::MobileEntity*>(entity)) {
        // Register our training region boundary with the entity so that we will get events if they enter/exit
        mobileEntity->registerGeometryBoundary(bot_.selfState().trainingAreaGeometry->clone(), bot_.eventBroker());
      }
    } else if (const auto *entityMovementBegan = dynamic_cast<const event::EntityMovementBegan*>(event)) {
      if (walkingTargetAndAttack_ && entityMovementBegan->globalId == walkingTargetAndAttack_->targetId) {
        LOG() << "The target that we're walking to has changed its movement, we'll need to recalculate" << std::endl;
        walkingTargetAndAttack_.reset();
      }
    } else if (const auto *entityLifeStateChanged = dynamic_cast<const event::EntityLifeStateChanged*>(event)) {
      if (entityLifeStateChanged->globalId == bot_.selfState().globalId) {
        if (bot_.selfState().lifeState == sro::entity::LifeState::kDead) {
          if (childState_) {
            childState_.reset();
          }
        }
      }
    }
  }
  // Mark this event as handled, so that if we recurse, we dont handle it again
  RAII raii(handledEvents_, event);

  if (childState_) {
    // Either casting a skill, picking up an item ourself, picking up an item with COS, or walking somewhere.
    childState_->onUpdate(event);
    if (childState_->done()) {
      if (dynamic_cast<Walking*>(childState_.get()) != nullptr) {
        // Done walking.
        LOG() << "Done walking, current pos " << bot_.selfState().position() << std::endl;
      }
      childState_.reset();
    } else {
      // Child state is not yet done.
      if (dynamic_cast<Walking*>(childState_.get()) == nullptr) {
        // The Walking state machine is always interruptable. For every other state machine, we have nothing else to do in this function.
        return;
      }
    }
  }

  // We either have no child state, or we're walking somewhere.
  if (!bot_.selfState().spawned()) {
    LOG() << "We are not spawned, nothing to do" << std::endl;
    return;
  }

  if (bot_.selfState().lifeState == sro::entity::LifeState::kDead) {
    // Dead, nothing to do.
    // TODO: Maybe go to town after some time
    return;
  }

  if (bot_.needToGoToTown()) {
    done_ = true;
    return;
  }

  if (bot_.selfState().stunnedFromKnockback || bot_.selfState().stunnedFromKnockdown) {
    LOG() << "In Training and stunned from KB/KD" << std::endl;
  }

  // First, check that our buffs are all active. Use different sets of buffs for inside and outside of the training area.
  SkillList *buffList;
  if (trainingAreaGeometry_->pointIsInside(bot_.selfState().position())) {
    buffList = &trainingBuffs_;
  } else {
    buffList = &nonTrainingBuffs_;
  }

  bool setNewChildState = checkBuffs(*buffList);
  if (setNewChildState) {
    LOG() << "Have some buff to cast" << std::endl;
    onUpdate(event);
    return;
  }

  // TODO: Maybe we should have an event for when we enter the training area. That would trigger training area buffs immediately once we enter.
  // Buffs are all good at this point. First, check if we need to walk to the training area.
  if (!trainingAreaGeometry_->pointIsInside(bot_.selfState().position())) {
    // We are not in the training area.
    // Are we already walking there?
    if (childState_ && dynamic_cast<Walking*>(childState_.get()) != nullptr) {
      // We are already walking there. Nothing else to do.
      // TODO: We assume that this Walking is to the training area.
      return;
    } else if (childState_) {
      throw std::runtime_error("We're not in the training area, not walking to it, all buffs are good, but we still have some child state?");
    }

    // Walk to the training area
    const auto *trainingAreaCircle = dynamic_cast<const entity::Circle*>(trainingAreaGeometry_.get());
    if (trainingAreaCircle == nullptr) {
      throw std::runtime_error("Not sure where to navigate to for other shapes");
    }
    LOG() << "Not in training area, navigating to the center of the training area circle: " << trainingAreaCircle->center() << std::endl;
    setChildStateMachine<Walking>(trainingAreaCircle->center());
    onUpdate(event);
    return;
  }

  // We're in the training area.
  const auto [itemsInRange, monstersInRange] = getItemsAndMonstersInRange();

  setNewChildState = tryPickItem(itemsInRange);
  if (setNewChildState) {
    onUpdate(event);
    return;
  }

  setNewChildState = tryAttackMonster(monstersInRange);
  if (setNewChildState) {
    onUpdate(event);
    return;
  }

  // Nothing left to do except explore the training area. Randomly walk to a point.
  if (childState_ && dynamic_cast<Walking*>(childState_.get()) != nullptr) {
    // Already walking somewhere.
    return;
  }

  setNewChildState = walkToRandomPoint();
  if (setNewChildState) {
    onUpdate(event);
    return;
  }
  LOG() << "Made it to the end" << std::endl;
}

bool Training::tryPickItem(const ItemList &itemList) {
  if (itemList.empty()) {
    // Nothing in range to pick up.
    return false;
  }

  // Maybe going to pick up an item, check if we're even in a state to be picking up items
  std::optional<sro::scalar_types::EntityGlobalId> cosGlobalId;
  if (!bot_.selfState().cosInventoryMap.empty()) {
    // Have a cos
    // TODO: How do we pick which COS to use?
    //  Maybe there can only ever be one.
    //  For now, just use the "first"
    cosGlobalId = bot_.selfState().cosInventoryMap.begin()->first;
  }
  if (!cosGlobalId && !canMove()) {
    LOG() << "No way to pick up items. No pickpet and cant move." << std::endl;
    return false;
  }

  // Have some option to pick item
  // Lets see if there's anything to pick up
  auto wantItem = [](const entity::Item *item) {
    // TODO: Check if this item should be picked up, based on a configured filter
    return (item->refObjId != 62 && // Arrows
            item->refObjId != 10383); // Bolts
  };
  for (const auto *item : itemList) {
    if (!item->ownerJId || *item->ownerJId == bot_.selfState().jId) {
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
        possiblyOverwriteChildStateMachine<PickItem>(targetItemGlobalId);
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
    LOG() << "No target/skill chosen" << std::endl;
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
  const auto &targetEntity = bot_.worldState().getEntity<entity::MobileEntity>(targetAndAttack->targetId);
  const auto attackRefId = targetAndAttack->skillId;
  std::optional<sro::Position> destinationPosition;
  if (!finishedWalking) {
    destinationPosition = calculateWhereToWalkToAttackEntityWithSkill(targetEntity, attackRefId);
  }

  if (destinationPosition) {
    // Need to walk to cast skill on entity.
    // LOG() << "Need to walk to " << *destinationPosition << " to cast skill on entity, which is " << sro::position_math::calculateDistance2d(targetEntity.position(), *destinationPosition) << " away from the target" << std::endl;
    possiblyOverwriteChildStateMachine<Walking>(*destinationPosition, true);
    walkingTargetAndAttack_ = targetAndAttack;
  } else {
    // We are within range to cast skill.
    CastSkillStateMachineBuilder castSkillBuilder(bot_, attackRefId);
    castSkillBuilder.withTarget(targetEntity.globalId);
    const auto &skillData = bot_.gameData().skillData().getSkillById(attackRefId);
    const auto weaponSlot = getInventorySlotOfWeaponForSkill(skillData, bot_);
    if (weaponSlot) {
      castSkillBuilder.withWeapon(*weaponSlot);
    }
    if (imbueRefId_) {
      castSkillBuilder.withImbue(*imbueRefId_);
    }
    possiblyOverwriteChildStateMachine(castSkillBuilder.create());
  }
  return true;
}

bool Training::walkToRandomPoint() {
  if (!canMove()) {
    return false;
  }

  if (bot_.selfState().moving()) {
    LOG() << "We're not in charge of a movement, but we're moving somewhere. What's this?" << std::endl;
  }

  const auto destPos = packet::building::NetworkReadyPosition(randomPointInGeometry(trainingAreaGeometry_.get()));
  setChildStateMachine<Walking>(destPos.asSroPosition(), false);
  return true;
}

std::tuple<Training::ItemList, Training::MonsterList> Training::getItemsAndMonstersInRange() const {
  ItemList itemsInRange;
  MonsterList monstersInRange;
  for (const auto &entityIdPtrPair : bot_.entityTracker().getEntityMap()) {
    if (const auto *monster = dynamic_cast<const entity::Monster*>(entityIdPtrPair.second.get())) {
      if (monster->lifeState != sro::entity::LifeState::kAlive) {
        // Dont care about monsters that arent alive
        continue;
      }
      if (monster->knowCurrentHp() && monster->currentHp() == 0) {
        // Monster isnt dead, but has 0 hp. It's effectively dead to us.
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
    LOG() << "Next buff to cast is " << *nextBuffToCast << std::endl;
    const auto &buffData = bot_.gameData().skillData().getSkillById(*nextBuffToCast);

    // Does the buff require a specific weapon to be equipped to cast?
    const auto weaponSlot = getInventorySlotOfWeaponForSkill(buffData, bot_);
    // Note: It is also possible that a skill requires a shield (shield bash)
    if (weaponSlot) {
      LOG() << "Inventory slot of weapon for skill is " << static_cast<int>(*weaponSlot) << std::endl;
      castSkillBuilder.withWeapon(*weaponSlot);
    }

    // TODO: Shield too?

    if (buffData.targetRequired) {
      // TODO: We assume this buff is for ourself
      LOG() << "Buff requires a target, using self (req self? " << buffData.targetGroupSelf << ')' << std::endl;
      castSkillBuilder.withTarget(bot_.selfState().globalId);
    }

    LOG() << "Created child state to cast skill" << std::endl;
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
  const auto selfCurrentPosition = bot_.selfState().position();
  const auto &skill = bot_.gameData().skillData().getSkillById(attackRefId);
  const double skillRangeMinusRounding = std::max(sro::constants::kSqrtHalf, skill.actionRange - sro::constants::kSqrtHalf*2);
  const auto calcd_dist = sro::position_math::calculateDistance2d(selfCurrentPosition, entity.position());
  if (calcd_dist <= skillRangeMinusRounding) {
    LOG() << "Already close enough" << std::endl;
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
      const auto timeToGetToCurrentDest = distanceToDestination / bot_.selfState().currentSpeed() + kEstimatedPingTimeSeconds;

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

bool Training::done() const {
  return done_;
}

bool Training::wantToAttackMonster(const entity::Monster &monster) const {
  const auto &characterData = bot_.gameData().characterData().getCharacterById(monster.refObjId);

  // TODO: Move this to the config, or determine at runtime.
  // if (std::abs(characterData.lvl - bot_.selfState().getCurrentLevel()) > 10) {
  //   // Don't attack anything 10 levels above or below us
  //   return false;
  // }

  return trainingAreaGeometry_->pointIsInside(monster.position());
}

void Training::buildBuffList() {
  // TODO: This data should come from some config
  imbueRefId_ = 8129; // 8129, "Thunder Phoenix Force"

  // Create a list of buffs to use
  trainingBuffs_ = SkillList {
      8150, // Ghost Walk - God
      8115, // Snow Shield - Intensify
      8133, // God - Piercing Force
      7980, // Final Guard of Ice
      8183, // Concentration - 4th

      // Cleric
      // 11795, // Holy Recovery Division
      // 11934, // Holy Spell

      // Rogue
      // 9516, // Crossbow Extreme
    };

  nonTrainingBuffs_ = SkillList {
      8150, // Ghost Walk - God
    };

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
        LOG() << "Moving buffs around. Moving " << i << " to " << backIndex << std::endl;
        std::swap(buffList[i], buffList[backIndex]);
        --backIndex;
      }
    }
  };

  sortBuffs(trainingBuffs_);
  sortBuffs(nonTrainingBuffs_);
}

std::optional<sro::scalar_types::ReferenceObjectId> Training::getNextBuffToCast(const SkillList &buffList) const {
  // Lets evaluate our buffs and see if any need to be reactivated
  auto copyOfBuffsToUse = buffList;
  // Remove all buffs which are currently active.
  copyOfBuffsToUse.erase(std::remove_if(copyOfBuffsToUse.begin(), copyOfBuffsToUse.end(), [&](const auto &buff) {
    return bot_.selfState().buffIsActive(buff);
  }), copyOfBuffsToUse.end());

  if (copyOfBuffsToUse.empty()) {
    // No buffs need to be cast
    return {};
  }

  // Choose one that isnt active, on cooldown, or already used
  for (int i=0; i<copyOfBuffsToUse.size(); ++i) {
    const auto buff = copyOfBuffsToUse.at(i);
    if (bot_.selfState().skillEngine.alreadyTriedToCastSkill(buff)) {
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
    LOG() << "Want to attack a monster but have no available skills" << std::endl;
    return {};
  }

  struct RarityAndAttacking {
    // RarityAndAttacking(sro::entity::MonsterRarity r, bool a) : rarity(r), isAttacking(a) {}
    sro::entity::MonsterRarity rarity;
    bool isAttacking;
  };

  // Lower number is higher priority
  auto getPriority = [this](const entity::Monster *monster) {
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
    bool isAttackingUs = (monster->targetGlobalId && *monster->targetGlobalId == bot_.selfState().globalId);
    int index=0;
    for (const auto &i : monsterPriorities) {
      if (i.rarity == monster->rarity && i.isAttacking == isAttackingUs) {
        return index;
      }
      ++index;
    }
    LOG() << "Dont know prioritiy of this monster!" << std::endl;
    return 100;
  };

  // Choose the highest priority monster (lowest priority value, tiebreak with which is closer)
  const auto currentPosition = bot_.selfState().position();
  auto min_it = std::min_element(monsters.begin(), monsters.end(), [&currentPosition, &getPriority](const entity::Monster *lhs, const entity::Monster *rhs) {
    const auto lhsPriority = getPriority(lhs);
    const auto rhsPriority = getPriority(rhs);
    if (lhsPriority == rhsPriority) {
      // If same priority, closer is better
      return sro::position_math::calculateDistance2d(currentPosition, lhs->position()) < sro::position_math::calculateDistance2d(currentPosition, rhs->position());
    } else {
      return lhsPriority < rhsPriority;
    }
  });
  // Choose the first attack
  const auto *targetMonster = dynamic_cast<const entity::Monster*>(*min_it);
  if (targetMonster == nullptr) {
    throw std::runtime_error("Target is not a monster");
  }
  return TargetAndAttackSkill{targetMonster->globalId, availableSkills.front()};
}

} // namespace state::machine