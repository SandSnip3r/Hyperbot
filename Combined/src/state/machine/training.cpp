#include "castSkill.hpp"
#include "castSkillOnEntity.hpp"
#include "training.hpp"
#include "walking.hpp"

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
  skillsToUse_ = std::vector<sro::scalar_types::ReferenceObjectId> {
    // TODO: Why do i get a Handle Read Error when I send an invalid skill?
      8204, // Crane's Thunderbolt
      7805, // Flying Dragon - Flash
      8195, // Horse's Thunderbolt

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
  // Observe incoming events, regardless if we have a child

  // Did something just change that will trigger us to go back to town?
  //  What are the possible reasons that we'd want to go to town?
  //    Durability is too low
  //    Too few HP potions remaining
  //    Too few MP potions remaining
  //    Too few <misc expendable item> remaining
  //    Quest completed
  //    Inventory full (or close)
  //    Collected a sufficient number of some item (? cannot think of concrete example that is different from quest items)
  //    Dead
  //  This boils down to the following events:
  //    Durability of item changed
  //    Inventory item updated (quantity decreased or new item picked)
  //    Died

  // TODO: Improve on this mechanism
  // When we recurse, we do not want to handle the same event again.
  // If we passed nullptr as our event, we would solve this problem, but we do want to pass that event to the presumably newly created child state machine.
  // So, instead, we keep track of the events we've already handled, and only handle the event if we havent yet handled it.
  // This problem is currently only unique to this state machine because this is the only one which does some event processing before forwarding the event to the child state machine.
  struct RAII {
    RAII(std::set<const event::Event*> &handledEvents) : handledEvents_(handledEvents) {}
    std::set<const event::Event*> &handledEvents_;
    const event::Event *event_{nullptr};
    void setEvent(const event::Event *event) {
      event_ = event;
      handledEvents_.emplace(event_);
    }
    ~RAII() {
      if (event_ != nullptr) {
        handledEvents_.erase(event_);
      }
    }
  };
  RAII raii(handledEvents_);

  if (event != nullptr && (handledEvents_.find(event) == handledEvents_.end())) {
    // Received some specific event info
    if (const auto *entitySpawnedEvent = dynamic_cast<const event::EntitySpawned*>(event)) {
      auto *entity = bot_.entityTracker().getEntity(entitySpawnedEvent->globalId);
      if (auto *mobileEntity = dynamic_cast<entity::MobileEntity*>(entity)) {
        // Register our training region boundary with the entity so that we will get events if they enter/exit
        mobileEntity->registerGeometryBoundary(bot_.selfState().trainingAreaGeometry->clone(), bot_.eventBroker());
      }
    } else if (const auto *inventoryUpdatedEvent = dynamic_cast<const event::InventoryUpdated*>(event)) {
      // Maybe we have too few items now?
    } else if (const auto *inventoryItemUpdatedEvent = dynamic_cast<const event::InventoryItemUpdated*>(event)) {
      // Maybe durability is too low?
    } else if (const auto *entityLifeStateChanged = dynamic_cast<const event::EntityLifeStateChanged*>(event)) {
      // Maybe we died?
    }

    // Mark this event as handled, so that if we recurse, we dont handle it again
    raii.setEvent(event);
  }

  if (childState_) {
    // Have a child state, it takes priority
    childState_->onUpdate(event);
    if (childState_->done()) {
      childState_.reset();
    } else {
      // Dont execute anything else in this function until the child state is done
      return;
    }
  }

  if (!bot_.selfState().spawned()) {
    LOG() << "We are not spawned, nothing to do" << std::endl;
  }

  if (bot_.needToGoToTown()) {
    done_ = true;
    return;
  }

  if (bot_.selfState().stunnedFromKnockback || bot_.selfState().stunnedFromKnockdown) {
    LOG() << "In Training and stunned from KB/KD" << std::endl;
  }

  // Try to cast buffs before even looking at what's around us
  const auto nextBuffToCast = getNextBuffToCast();
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
    setChildStateMachine(castSkillBuilder.create());
    onUpdate(event);
    return;
  }

  // All buffs are active.
  // TODO: Differentiate between buffs used while walking to the training area vs buffs used while training.
  // If we're not in the training area, go there.
  if (!trainingAreaGeometry_->pointIsInside(bot_.selfState().position())) {
    const auto *trainingAreaCircle = dynamic_cast<const entity::Circle*>(trainingAreaGeometry_.get());
    if (trainingAreaCircle == nullptr) {
      throw std::runtime_error("Not sure where to navigate to for other shapes");
    }
    LOG() << "Not in training area, navigating to the center of the training area circle: " << trainingAreaCircle->center() << std::endl;
    setChildStateMachine<Walking>(bot_, trainingAreaCircle->center());
    onUpdate(event);
    return;
  }

  // What's in the area?
  std::vector<const entity::Monster*> monstersInRange;
  std::vector<const entity::Item*> itemsInRange;
  int totalMonsterCount{0};
  int totalItemCount{0};
  for (const auto &entityIdPtrPair : bot_.entityTracker().getEntityMap()) {
    if (const auto *monster = dynamic_cast<const entity::Monster*>(entityIdPtrPair.second.get())) {
      if (monster->lifeState != sro::entity::LifeState::kAlive) {
        // Dont care about monsters that arent alive
        continue;
      }
      if (monster->knowCurrentHp() && monster->currentHp() == 0) {
        // Monster is effectively dead
        LOG() << "Monster isnt dead, but has 0 hp" << std::endl;
        continue;
      }
      ++totalMonsterCount;
      if (wantToAttackMonster(*monster)) {
        monstersInRange.push_back(monster);
      }
    } else if (const auto *item = dynamic_cast<const entity::Item*>(entityIdPtrPair.second.get())) {
      ++totalItemCount;
      if (trainingAreaGeometry_->pointIsInside(item->position())) {
        itemsInRange.push_back(item);
      }
    }
  }
  if (!monstersInRange.empty()) {
    LOG() << "There are " << monstersInRange.size() << " monsters in range /" << totalMonsterCount << std::endl;
  }
  if (!itemsInRange.empty()) {
    LOG() << "There are " << itemsInRange.size() << " items in range /" << totalItemCount << std::endl;
  }

  if (!itemsInRange.empty()) {
    // Maybe going to pick up an item, check if we're even in a state to be picking up items
    std::optional<sro::scalar_types::EntityGlobalId> cosGlobalId;
    if (!bot_.selfState().cosInventoryMap.empty()) {
      // Have a cos
      // TODO: How do we pick which COS to use?
      //  Maybe there can only ever be one.
      //  For now, just use the "first"
      cosGlobalId = bot_.selfState().cosInventoryMap.begin()->first;
    }
    auto wantItem = [](const entity::Item *item) {
      // TODO: Check if this item should be picked up, based on a configured filter
      return item->refObjId != 62; // Dont pick arrows
    };
    if (cosGlobalId || canMove()) {
      // Have some option to pick item
      // Lets see if there's anything to pick up
      for (const auto *item : itemsInRange) {
        if (!item->ownerJId || *item->ownerJId == bot_.selfState().jId) {
          // No owner or it belongs to us. Pick it up
          // TODO: Track party members' JIDs so that we know if we can pick up their items.
          if (!wantItem(item)) {
            // Skipping item
            continue;
          }
          const auto targetItemGlobalId = item->globalId;
          if (cosGlobalId) {
            setChildStateMachine<PickItemWithCos>(bot_, *cosGlobalId, targetItemGlobalId);
          } else {
            LOG() << "Going to pick item" << std::endl;
            setChildStateMachine<PickItem>(bot_, targetItemGlobalId);
          }
          onUpdate(event);
          return;
        }
      }
    } else {
      LOG() << "No way to pick up items. No pickpet and cant move" << std::endl;
    }
  }

  if (!monstersInRange.empty()) {
    // Figure out which skills we have available to us
    std::vector<sro::scalar_types::ReferenceObjectId> availableSkills;
    for (const auto skillId : skillsToUse_) {
      if (bot_.canCastSkill(skillId)) {
        availableSkills.push_back(skillId);
      }
    }
    LOG() << "Trying to attack a monster; we have " << availableSkills.size() << " available skill(s)" << std::endl;
    if (availableSkills.empty()) {
      // No available skills
      LOG() << "No available skills" << std::endl;
    } else {
      const auto [target, attackRefId] = getTargetAndAttackSkill(monstersInRange, availableSkills);
      const auto destinationPosition = calculateWhereToWalkToAttackEntityWithSkill(target, attackRefId);
      // TODO: Combine the two cases below.
      if (destinationPosition) {
        // Need to walk to cast skill on entity.
        setChildStateMachine<CastSkillOnEntity>(bot_, attackRefId, target->globalId, *destinationPosition);
      } else {
        // We are within range to cast skill.
<<<<<<< HEAD
        CastSkillStateMachineBuilder castSkillBuilder(bot_, attackRefId);
=======
        CastSkillStateMachineBuilder castSkillBuilder(attackRefId);
>>>>>>> a1a120f3f74f130270fbc95b0d2624a977ae5e8f
        castSkillBuilder.withTarget(target->globalId);
        const auto &skillData = bot_.gameData().skillData().getSkillById(attackRefId);
        const auto weaponSlot = getInventorySlotOfWeaponForSkill(skillData, bot_);
        if (weaponSlot) {
          castSkillBuilder.withWeapon(*weaponSlot);
        }
        setChildStateMachine(castSkillBuilder.create());
      }
      onUpdate(event);
      return;
    }
  } else if (canMove()) {
    // Nothing to attack, lets just move to a random location inside the training area
    if (bot_.selfState().moving()) {
      // Already walking somewhere, dont interrupt
      return;
    }
    const auto destPos = packet::building::NetworkReadyPosition(randomPointInGeometry(trainingAreaGeometry_.get()));
    LOG() << "Walking to random point. From " << bot_.selfState().position() << " to " << destPos.asSroPosition() << std::endl;
    const auto movementPacket = packet::building::ClientAgentCharacterMoveRequest::moveToPosition(destPos);
    bot_.packetBroker().injectPacket(movementPacket, PacketContainer::Direction::kClientToServer);
  } else {
    LOG() << "Nothing to attack and we cant move right now" << std::endl;
  }
}

std::optional<sro::Position> Training::calculateWhereToWalkToAttackEntityWithSkill(const entity::MobileEntity *entity, sro::scalar_types::ReferenceObjectId attackRefId) {
  const auto selfCurrentPosition = bot_.selfState().position();
  const auto &skill = bot_.gameData().skillData().getSkillById(attackRefId);
  const double skillRangeMinusRounding = std::max(0.0, skill.actionRange - sro::constants::kSqrtHalf);
  if (sro::position_math::calculateDistance2d(selfCurrentPosition, entity->position()) <= skillRangeMinusRounding) {
    // We are already close enough.
    return {};
  }

  auto calculateDestination = [&](const sro::Position &targetPos) {
    // We don't need to walk completely to the position of the target, instead, we will take the path to the target, and find the earliest point on that path which is within range of the target.
    const auto pathToDestination = calculatePathToDestination(targetPos, bot_);
    if (pathToDestination.size() < 2) {
      throw std::runtime_error("Path to target has fewer than two points. We expect this path to at least contain the starting pos and end pos.");
    }
    // Find the last point in the path which is outside of range of the target.
    int index = -1;
    for (int i=pathToDestination.size()-2; i>=0; --i) {
      const auto pos = pathToDestination.at(i).asSroPosition();
      if (sro::position_math::calculateDistance2d(targetPos, pos) > skillRangeMinusRounding) {
        // This point is outside of range of the target.
        index = i;
        break;
      }
    }
    if (index == -1) {
      throw std::runtime_error("No point on the path it outside of range of the target.");
    }
    if (index < pathToDestination.size()-2) {
      // There are multiple segments which are inside the radius, this probably means we're avoiding an obstacle
      // TODO: We should pick the second to last position, since that is probably the first position which has direct line of sight to the target.
      LOG() << std::string(600,' ') << "\nMultiple segments within range of target!!!!\n" << std::string(600,' ') << std::endl;
    }
    // Find where the line pathToDestination[index] -> pathToDestination[index+1] intersects with the circle defined around the target with the radius as the range of the skill.
    // First, convert both points to pathfinder::Vectors.

    const auto [res0X, res0Z] = sro::position_math::calculateOffsetInOtherRegion(pathToDestination.at(index).asSroPosition(), targetPos);
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

  auto destinationPos = calculateDestination(entity->position());
  if (entity->moving()) {
    // Run an iterative algorithm to walk to where the target will be by the time we get there.
    int remainingRecalculationCount = 10;
    const float kDistanceTolerance = 0.01;
    while (1) {
      // We already calculated a destination that we want to walk to.
      // First calculate how long it will take us to get there.
      const auto distanceToDestination = sro::position_math::calculateDistance2d(selfCurrentPosition, destinationPos);
      const auto timeToGetToCurrentDest = distanceToDestination / bot_.selfState().currentSpeed();

      // Next, calculate where the target will be after that amount of time.
      const auto targetNewPos = entity->positionAfterTime(timeToGetToCurrentDest);

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
  // Create a list of buffs to use
  buffsToUse_ = std::vector<sro::scalar_types::ReferenceObjectId> {
      8150, // Ghost Walk - God
      8133, // God - Piercing Force
      7980, // Final Guard of Ice
      8183, // Concentration - 4th

      // Cleric
      // 11795, // Holy Recovery Division
      // 11934, // Holy Spell

      // Rogue
      // 9516, // Crossbow Extreme
    };

  // Move all buffs which require a weapon at all times to the end
  // TODO: Can do better, move skills which require a weapon at all times AND have a cooldown shorter than the skill duration even further to the back of the list
  int backIndex = buffsToUse_.size()-1;
  for (int i=0; i<backIndex; ++i) {
    const auto &buffRefId = buffsToUse_[i];
    const auto &buffData = bot_.gameData().skillData().getSkillById(buffRefId);
    const auto buffRequiredWeapons = buffData.reqi();
    if (!buffRequiredWeapons.empty()) {
      // Buff requires a weapon at all times
      LOG() << "Moving buffs around. Moving " << i << " to " << backIndex << std::endl;
      std::swap(buffsToUse_[i], buffsToUse_[backIndex]);
      --backIndex;
    }
  }
}

std::optional<sro::scalar_types::ReferenceObjectId> Training::getNextBuffToCast() const {
  // Lets evaluate our buffs and see if any need to be reactivated
  auto copyOfBuffsToUse = buffsToUse_;
  // Calculate a diff between what's active and what we want to be active
  for (const auto buff : bot_.selfState().buffs) {
    if (auto it = std::find(copyOfBuffsToUse.begin(), copyOfBuffsToUse.end(), buff); it != copyOfBuffsToUse.end()) {
      // This buff is already active, remove from list
      copyOfBuffsToUse.erase(it);
    }
  }

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

// TODO: Also return the position to move to, to cast the skill on the target
std::pair<const entity::Monster*, sro::scalar_types::ReferenceObjectId> Training::getTargetAndAttackSkill(const std::vector<const entity::Monster*> &monsters, const std::vector<sro::scalar_types::ReferenceObjectId> &attackSkills) const {
  if (monsters.empty()) {
    throw std::runtime_error("monsters empty");
  }
  if (attackSkills.empty()) {
    throw std::runtime_error("attackSkills empty");
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
  return {*min_it, attackSkills.front()};
}

} // namespace state::machine