#ifndef MODULE_SKILL_USE_MODULE_HPP_
#define MODULE_SKILL_USE_MODULE_HPP_

#include "../broker/eventBroker.hpp"
#include "../broker/packetBroker.hpp"
#include "../packet/building/clientAgentActionCommandRequest.hpp"
#include "../packet/parsing/packetParser.hpp"
#include "../packet/parsing/clientAgentActionCommandRequest.hpp"
#include "../packet/parsing/clientAgentChatRequest.hpp"
#include "../packet/parsing/serverAgentActionCommandResponse.hpp"
#include "../packet/parsing/serverAgentActionSelectResponse.hpp"
#include "../packet/parsing/serverAgentBuffAdd.hpp"
#include "../packet/parsing/serverAgentBuffRemove.hpp"
#include "../packet/parsing/serverAgentChatUpdate.hpp"
#include "../packet/parsing/serverAgentEntityUpdateState.hpp"
#include "../packet/parsing/serverAgentSkillBegin.hpp"
#include "../packet/parsing/serverAgentSkillEnd.hpp"
#include "../pk2/gameData.hpp"
#include "../shared/silkroad_security.h"
#include "../state/entity.hpp"
#include "../state/self.hpp"
#include "../storage/storage.hpp"

#include <deque>
#include <mutex>
#include <optional>
#include <set>

namespace module {

// struct Skill {
//   uint32_t skillRefId;
//   std::optional<uint32_t> targetGId;
//   Skill(uint32_t skillId) : skillRefId(skillId) {}
//   Skill(uint32_t skillId, uint32_t targetId) : skillRefId(skillId), targetGId(targetId) {}
// };

struct Buff {
  Buff(uint32_t skillId, uint32_t t) : skillRefId(skillId), token(t) {}
  uint32_t skillRefId, token;
};

class SkillUseModule {
public:
  SkillUseModule(state::Entity &entityState,
                 state::Self &selfState,
                 storage::Storage &inventory,
                 broker::PacketBroker &brokerSystem,
                 broker::EventBroker &eventBroker,
                 const packet::parsing::PacketParser &packetParser,
                 const pk2::GameData &gameData);
private:
  state::Entity &entityState_;
  state::Self &selfState_;
  storage::Storage &inventory_;
  broker::PacketBroker &broker_;
  broker::EventBroker &eventBroker_;
  const packet::parsing::PacketParser &packetParser_;
  const pk2::GameData &gameData_;
  std::mutex contentionProtectionMutex_;

  // State
  uint32_t stateBitmask_ = 0;
  bool attackMode_ = false;
  std::optional<int32_t> targetGId_;
  // =======Current=======
  static const int kKnockbackStatusDurationMs = 2000;
  std::optional<broker::TimerManager::TimerId> knockedBackCooldownEventId_;

  std::deque<packet::structures::ActionCommand> actionsRequestedToBeQueued_;
  std::deque<packet::structures::ActionCommand> queuedActions_;
  std::set<uint32_t> skillsOnCooldown_;
  // std::optional<packet::structures::ActionCommand> activeAction_, queuedAction_;
  // std::optional<packet::structures::ActionCommand> pendingCommonAttack_, queuedCommonAttack_;
  // std::optional<int32_t> activeChainSkillId_;
  // =====================

  std::map<uint32_t, uint32_t> castSkillMap_;
  // std::set<uint32_t> queuedInstantSkills_;

  // Skills
  // const uint32_t kSkillPhysicalDefense = 0x2ED1; // Phys def
  // const uint32_t kSkillMagicalDefense = 0x2EDF; // Mag def
  // const uint32_t kSkillHolyRecoveryDivision = 0x2E15; // Holy recovery division
  // const uint32_t kSkillHolySpell = 0x2E9F; // Holy spell
  // const uint32_t kSkillHealingCycle = 0x2DE0; // Healing cycle
  // const uint32_t kSkillHolyRecovery = 0x2E07; // Holy recovery (heal)
  // const uint32_t kSkillHolyGroupRecovery = 0x2E23; // Holy group recovery (heal)
  // const uint32_t kSkillManaCycle = 0x7625; // Mana cycle
  // const uint32_t kSkillDaggerDesperate = 0x25E8; // Dagger desperate
  // const uint32_t kSkillGroupRecovery = 0x2E1D; // Group Recovery (ON SAME COOLDOWN AS HOLY GROUP RECOVERY)
  // const uint32_t kSkillGroupHealingBreath = 0x2DBF; // Group Healing Breath
  // const uint32_t kSkillGroupHealing = 11702;

  // Active buffs
  // bool waitingForCast_ = false;
  // std::vector<uint32_t> skillsToUse_ = {
  //                                       //  8097, // Snow Shield Novice
  //                                        8244, // Mag def
  //                                       //  7985, // Phys def
  //                                        8186, // Parry
  //                                        8135, // Mag attack
  //                                        8223, // Phys attack
  //                                        7885, // Fire Hawk
  //                                        7910, // Range increase
  //                                        8155, // Speed
  //                                        8217, // Imbue
  //                                       //  7865, // 7 arrow combo
  //                                       //  7862, // 6 arrow combo
  //                                        7941, // Strong Bow Destruction
  //                                        7922, // Pitch Black Arrow
  //                                        7844, // Anti Devil Bow - Moon Light
  //                                        7843 // Anti Devil Bow - Demolition
  //                                      };
  std::vector<Buff> activeBuffs_;
  // std::deque<int32_t> tryCastSkills_;
  

  // Packet handling functions
  bool handlePacket(const PacketContainer &packet);
  bool clientAgentChatRequestReceived(packet::parsing::ClientAgentChatRequest &packet);
  void clientAgentActionCommandRequestReceived(packet::parsing::ClientAgentActionCommandRequest &packet);
  void serverAgentBuffAddReceived(packet::parsing::ServerAgentBuffAdd &packet);
  void serverAgentBuffRemoveReceived(packet::parsing::ServerAgentBuffRemove &packet);
  void serverAgentActionCommandResponseReceived(packet::parsing::ServerAgentActionCommandResponse &packet);
  void serverAgentActionSelectResponseReceived(packet::parsing::ServerAgentActionSelectResponse &packet);
  void serverAgentEntityUpdateStateReceived(packet::parsing::ServerAgentEntityUpdateState &packet);
  void serverAgentSkillBeginReceived(packet::parsing::ServerAgentSkillBegin &packet);
  void serverAgentSkillEndReceived(packet::parsing::ServerAgentSkillEnd &packet);
  void serverAgentChatUpdateReceived(packet::parsing::ServerAgentChatUpdate &packet);


  // Event handling functions
  void handleEvent(const event::Event *event);
  void attack();
  
  // General functions
  void printQueues() const;

  bool skillInQueue(uint32_t skillId);
  void activeActionStarted();
  void targetDied();
  void died();
  void knockedBack();
  void knockedDown();
  void tryCastNext();
  void requestedAction(const packet::structures::ActionCommand &actionCommand);
  void printBuffs();
  void useNextSkillInQueue();
  void selectEntity(state::Entity::EntityId entityId);
  void commonAttackEntity(state::Entity::EntityId entityId);
  void traceEntity(state::Entity::EntityId entityId);
  void pickupEntity(state::Entity::EntityId entityId);
};

} // namespace module

#endif // MODULE_SKILL_USE_MODULE_HPP_