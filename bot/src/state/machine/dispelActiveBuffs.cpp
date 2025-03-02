#include "dispelActiveBuffs.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentActionCommandRequest.hpp"

#include <absl/log/log.h>

namespace state::machine {

DispelActiveBuffs::DispelActiveBuffs(Bot &bot) : StateMachine(bot) {
}

DispelActiveBuffs::~DispelActiveBuffs() {
}

Status DispelActiveBuffs::onUpdate(const event::Event *event) {
  if (event != nullptr) {
    if (const auto *buffRemoved = dynamic_cast<const event::BuffRemoved*>(event); buffRemoved != nullptr) {
      if (buffRemoved->entityGlobalId == bot_.selfState()->globalId) {
        CHAR_VLOG(1) << bot_.selfState()->name << " has dispelled buff " << bot_.gameData().getSkillName(buffRemoved->buffRefId);
        if (buffSkillId_ && buffRemoved->buffRefId == *buffSkillId_) {
          buffSkillId_.reset();
          if (dispelTimeoutEventId_) {
            bot_.eventBroker().cancelDelayedEvent(*dispelTimeoutEventId_);
            dispelTimeoutEventId_.reset();
          }
        }
      }
    } else if (event->eventCode == event::EventCode::kTimeout) {
      if (dispelTimeoutEventId_ && event->eventId == *dispelTimeoutEventId_) {
        CHAR_VLOG(1) << "Dispelling buff timed out";
        dispelTimeoutEventId_.reset();
        buffSkillId_.reset();
      }
    }
  }

  if (buffSkillId_) {
    // Waiting on a buff to be dispelled.
    return Status::kNotDone;
  }

  // What active buffs do we have?
  const auto activeBuffs = bot_.selfState()->activeBuffs();
  if (activeBuffs.empty()) {
    CHAR_VLOG(1) << bot_.selfState()->name << " has no active buffs";
    return Status::kDone;
  }

  // If any buff has param "nbuf", it is a debuff which cannot be cancelled. Depending on the time remaining on the debuff, it might be faster to teleport to remove it.
  for (const auto skillId : activeBuffs) {
    const bool isDebuff = bot_.gameData().skillData().getSkillById(skillId).hasParam("nbuf");
    CHAR_VLOG(2) << bot_.selfState()->name << " has active buff " << bot_.gameData().getSkillName(skillId) << " with ID " << skillId << (isDebuff ? " (debuff)" : "");
    if (!isDebuff) {
      buffSkillId_ = skillId;
    }
  }
  if (buffSkillId_) {
    bot_.packetBroker().injectPacket(packet::building::ClientAgentActionCommandRequest::dispel(*buffSkillId_), PacketContainer::Direction::kClientToServer);
    dispelTimeoutEventId_ = bot_.eventBroker().publishDelayedEvent(std::chrono::milliseconds(1000), event::EventCode::kTimeout);
  } else {
    CHAR_VLOG(1) << bot_.selfState()->name << " has no active buffs that can be dispelled, but do have a debuff we must wait for";
  }
  return Status::kNotDone;
}

} // namespace state::machine
