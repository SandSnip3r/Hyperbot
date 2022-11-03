#include "castSkill.hpp"
#include "training.hpp"

#include "entity/geometry.hpp"
#include "bot.hpp"
#include "helpers.hpp"
#include "logging.hpp"
#include "packet/building/clientAgentCharacterMoveRequest.hpp"
#include "type_id/categories.hpp"

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

} // anonymous namespace

namespace state::machine {

Training::Training(Bot &bot, const sro::Position &trainingSpotCenter) : StateMachine(bot), trainingSpotCenter_(trainingSpotCenter) {
  buildBuffList();
  // Create a list of skills to use
  skillsToUse_ = std::vector<sro::scalar_types::ReferenceObjectId> {
      // 9612, // Distance Shot
      // 9631, // Hurricant Shot
      // 9544, // Intense Shot
      10264, // Fire Bolt
      // 10122, // Ice Bolt
      9587, // Rapid Shot
      8499, // Sprint Assault
      // 11261, // Weird Chord
      // 11072, // Vampire Kiss
    };

  bot_.selfState().setTrainingAreaGeometry(std::make_unique<entity::Circle>(trainingSpotCenter_, kMonsterRange_));
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
}

void Training::onUpdate(const event::Event *event) {
TODO_REMOVE_THIS_LABEL:
  if (childState_) {
    // Have a child state, it takes priority
    LOG() << "Going into child state" << std::endl;
    childState_->onUpdate(event);
    if (childState_->done()) {
      LOG() << "Child state is done" << std::endl;
      childState_.reset();
    } else {
      // Dont execute anything else in this function until the child state is done
      return;
    }
  }

  if (event != nullptr) {
    // Received some specific event info
    if (const auto *entitySpawnedEvent = dynamic_cast<const event::EntitySpawned*>(event)) {
      auto *entity = bot_.entityTracker().getEntity(entitySpawnedEvent->globalId);
      if (auto *mobileEntity = dynamic_cast<entity::MobileEntity*>(entity)) {
        // Register our training region boundary with the entity so that we will get events if they enter/exit
        mobileEntity->registerGeometryBoundary(bot_.selfState().trainingAreaGeometry->clone(), bot_.eventBroker());
      }
    }
  }

  // Try to cast buffs before even looking at what's around us
  const auto nextBuffToCast = getNextBuffToCast();
  if (nextBuffToCast) {
    auto castSkillBuilder = CastSkillStateMachineBuilder(bot_, *nextBuffToCast);
    LOG() << "Next buff to cast is " << *nextBuffToCast << std::endl;
    const auto &buffData = bot_.gameData().skillData().getSkillById(*nextBuffToCast);

    // Does the buff require a specific weapon to be equipped to cast?
    const auto weaponSlot = getInventorySlotOfWeaponForSkill(buffData);
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
    childState_ = castSkillBuilder.create();
    goto TODO_REMOVE_THIS_LABEL;
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
      const auto distance = sro::position_math::calculateDistance2d(trainingSpotCenter_, monster->position());
      if (distance < kMonsterRange_) {
        monstersInRange.push_back(monster);
      }
    } else if (const auto *item = dynamic_cast<const entity::Item*>(entityIdPtrPair.second.get())) {
      ++totalItemCount;
      const auto distance = sro::position_math::calculateDistance2d(trainingSpotCenter_, item->position());
      if (distance < kItemRange_) {
        itemsInRange.push_back(item);
      }
    }
  }
  LOG() << "There are " << monstersInRange.size() << " monsters in range /" << totalMonsterCount << std::endl;
  LOG() << "There are " << itemsInRange.size() << " items in range /" << totalItemCount << std::endl;

  if (!monstersInRange.empty()) {
    // Figure out which skills we have available to us
    std::vector<sro::scalar_types::ReferenceObjectId> availableSkills;
    for (const auto skillId : skillsToUse_) {
      if (bot_.selfState().skillsOnCooldown.find(skillId) == bot_.selfState().skillsOnCooldown.end()) {
        // Skill is not on cooldown
        availableSkills.push_back(skillId);
      }
    }
    LOG() << "Trying to attack one monster" << std::endl;
    LOG() << "We have " << availableSkills.size() << " skill(s) not on cooldown" << std::endl;
    PacketContainer attackPacket;
    if (availableSkills.empty()) {
      // No available skills
      LOG() << "No available skills, not using common attacks" << std::endl;
    } else {
      const auto [target, attackRefId] = getTargetAndAttackSkill(monstersInRange, availableSkills);
      auto castSkillBuilder = CastSkillStateMachineBuilder(bot_, attackRefId).withTarget(target->globalId);
      const auto &skillData = bot_.gameData().skillData().getSkillById(attackRefId);
      const auto weaponSlot = getInventorySlotOfWeaponForSkill(skillData);
      if (weaponSlot) {
        castSkillBuilder.withWeapon(*weaponSlot);
      }
      childState_ = castSkillBuilder.create();
      goto TODO_REMOVE_THIS_LABEL;
    }
  } else {
    // Nothing to attack, lets just move to a random location inside the training area
    // TODO: Check if we can already see the entire training area?
    // Pick a random point inside a square that encloses our training circle
    static auto eng = createRandomEngine();
    auto notInCircle = [](double x, double y, double radius) {
      return sqrt(x*x+y*y) > radius;
    };
    std::uniform_real_distribution<double> dist(-kMonsterRange_, kMonsterRange_);
    auto x = dist(eng);
    auto y = dist(eng);
    while (notInCircle(x, y, kMonsterRange_)) {
      x = dist(eng);
      y = dist(eng);
    }
    // Transform x,y to sro coordinate
    const auto destPos = sro::position_math::createNewPositionWith2dOffset(trainingSpotCenter_, x, y);
    const auto movementPacket = packet::building::ClientAgentCharacterMoveRequest::moveToPosition(destPos);
    bot_.packetBroker().injectPacket(movementPacket, PacketContainer::Direction::kClientToServer);
    LOG() << "Walking to random pos " << sro::position_math::calculateDistance2d(trainingSpotCenter_, destPos) << "m from center of training area" << std::endl;
  }
}

bool Training::done() const {
  return false;
}

void Training::buildBuffList() {
  // TODO: This data should come from some config
  // Create a list of buffs to use
  buffsToUse_ = std::vector<sro::scalar_types::ReferenceObjectId> {
       11795, // Holy Recovery Division
       11934, // Holy Spell
      //  9516, // Crossbow Extreme
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
  auto printBuffs = [&copyOfBuffsToUse](const std::string &msg) {
    LOG() << msg << ": [ ";
    for (auto i : copyOfBuffsToUse) {
      std::cout << i << ", ";
    }
    std::cout << "]" << std::endl;
  };
  printBuffs("List of all buffs to use");

  // Calculate a diff between what's active and what we want to be active
  for (const auto buff : bot_.selfState().buffs) {
    if (auto it = std::find(copyOfBuffsToUse.begin(), copyOfBuffsToUse.end(), buff); it != copyOfBuffsToUse.end()) {
      // This buff is already active, remove
      copyOfBuffsToUse.erase(it);
    }
  }
  printBuffs("Buffs-to-use after already-active removed");

  if (copyOfBuffsToUse.empty()) {
    // No buffs need to be cast
    return {};
  }

  // Choose one that isnt on cooldown
  for (int i=0; i<copyOfBuffsToUse.size(); ++i) {
    const auto buff = copyOfBuffsToUse.at(i);
    if (bot_.selfState().skillsOnCooldown.find(buff) == bot_.selfState().skillsOnCooldown.end()) {
      return buff;
    } else {
      LOG() << "Buff " << buff << " is on cooldown, wont try to cast" << std::endl;
    }
  }
  LOG() << "All buffs are still on cooldown, none to cast" << std::endl;
  return {};
}

std::optional<uint8_t> Training::getInventorySlotOfWeaponForSkill(const pk2::ref::Skill &skillData) const {
  // TODO: Skill might not require a weapon
  const uint8_t kWeaponInventorySlot{6};
  std::vector<type_id::TypeCategory> possibleWeapons;
  for (auto i : skillData.reqi()) {
    if (i.typeId3 != 6) {
      LOG() << "reqi asks for non-weapon (typeId3: " << static_cast<int>(i.typeId3) << ")" << std::endl;
    }
    possibleWeapons.push_back(type_id::categories::kEquipment.subCategory(i.typeId3).subCategory(i.typeId4));
  }
  if (skillData.reqCastWeapon1 != 255) {
    possibleWeapons.push_back(type_id::categories::kWeapon.subCategory(skillData.reqCastWeapon1));
  }
  if (skillData.reqCastWeapon2 != 255) {
    possibleWeapons.push_back(type_id::categories::kWeapon.subCategory(skillData.reqCastWeapon2));
  }

  // First, check if the currently equipped weapon is valid for this skill
  if (bot_.selfState().inventory.hasItem(kWeaponInventorySlot)) {
    const auto *item = bot_.selfState().inventory.getItem(kWeaponInventorySlot);
    if (!item) {
      throw std::runtime_error("Have an item, but got null");
    }
    if (!type_id::categories::kWeapon.contains(item->typeData())) {
      throw std::runtime_error("Equipped \"weapon\" isnt a weapon");
    }
    if (item->isOneOf(possibleWeapons)) {
      // Currently equipped weapon can cast this skill
      return kWeaponInventorySlot;
    }
  }
  // Currently equipped weapon (if any) cannot cast this skill, search through our inventory for a weapon which can cast this skill
  std::vector<uint8_t> possibleWeaponSlots = bot_.selfState().inventory.findItemsOfCategory(possibleWeapons);
  LOG() << "Possible slots with weapon for skill: [ ";
  for (const auto slot : possibleWeaponSlots) {
    std::cout << static_cast<int>(slot) << ", ";
  }
  std::cout << "]" << std::endl;
  if (possibleWeaponSlots.empty()) {
    throw std::runtime_error("We have no weapon that can cast this skill");
  }

  // TODO: Pick best option
  // For now, pick the first option
  return possibleWeaponSlots.front();
}

std::pair<const entity::Monster*, sro::scalar_types::ReferenceObjectId> Training::getTargetAndAttackSkill(const std::vector<const entity::Monster*> &monsters, const std::vector<sro::scalar_types::ReferenceObjectId> &attackSkills) const {
  if (monsters.empty()) {
    throw std::runtime_error("monsters empty");
  }
  if (attackSkills.empty()) {
    throw std::runtime_error("attackSkills empty");
  }
  return {monsters.front(), attackSkills.front()};
}

} // namespace state::machine