#ifndef MODULE_SKILL_USE_MODULE_HPP_
#define MODULE_SKILL_USE_MODULE_HPP_

#include "../broker/eventBroker.hpp"
#include "../broker/packetBroker.hpp"
#include "../packet/building/clientAgentActionCommandRequest.hpp"
#include "../packet/parsing/packetParser.hpp"
#include "../packet/parsing/clientAgentChatRequest.hpp"
#include "../packet/parsing/serverAgentActionCommandResponse.hpp"
#include "../packet/parsing/serverAgentActionSelectResponse.hpp"
#include "../packet/parsing/serverAgentBuffAdd.hpp"
#include "../packet/parsing/serverAgentBuffRemove.hpp"
#include "../packet/parsing/serverAgentEntityUpdateState.hpp"
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

  // Skills
  const uint32_t kSkillPhysicalDefense = 0x2ED1; // Phys def
  const uint32_t kSkillMagicalDefense = 0x2EDF; // Mag def
  const uint32_t kSkillHolyRecoveryDivision = 0x2E15; // Holy recovery division
  const uint32_t kSkillHolySpell = 0x2E9F; // Holy spell
  const uint32_t kSkillHealingCycle = 0x2DE0; // Healing cycle
  const uint32_t kSkillHolyRecovery = 0x2E07; // Holy recovery (heal)
  const uint32_t kSkillHolyGroupRecovery = 0x2E23; // Holy group recovery (heal)
  const uint32_t kSkillManaCycle = 0x7625; // Mana cycle
  const uint32_t kSkillDaggerDesperate = 0x25E8; // Dagger desperate
  // std::vector<uint32_t> desiredBuffs
  const uint32_t kSkillGroupRecovery = 0x2E1D; // Group Recovery
  const uint32_t kSkillGroupHealingBreath = 0x2DBF; // Group Healing Breath

  // Active buffs
  bool tryingToBuff_ = false;
  std::vector<Buff> activeBuffs_;
  std::vector<uint32_t> desiredBuffs_ = { kSkillPhysicalDefense, kSkillMagicalDefense, kSkillHolyRecoveryDivision, kSkillHolySpell, kSkillHealingCycle, kSkillManaCycle, kSkillHolyRecovery, kSkillHolyGroupRecovery, kSkillGroupRecovery, kSkillGroupHealingBreath };
  std::set<uint32_t> skillsOnCooldown_;

  // Packet handling functions
  bool handlePacket(const PacketContainer &packet);
  void clientAgentBuffAddReceived(packet::parsing::ServerAgentBuffAdd &packet);
  void clientAgentBuffRemoveReceived(packet::parsing::ServerAgentBuffRemove &packet);
  bool clientAgentChatRequestReceived(packet::parsing::ClientAgentChatRequest &packet);
  void serverAgentActionCommandResponseReceived(packet::parsing::ServerAgentActionCommandResponse &packet);
  void serverAgentActionSelectResponseReceived(packet::parsing::ServerAgentActionSelectResponse &packet);
  void serverAgentEntityUpdateStateReceived(packet::parsing::ServerAgentEntityUpdateState &packet);

  // Event handling functions
  void handleEvent(const event::Event *event);
  void castBuffs();
  
  // General functions
  void weaponChanged();
  void preevaluateBuffs();
  void printBuffs();
  void useNextSkillInQueue();
  void selectEntity(state::Entity::EntityId entityId);
  void commonAttackEntity(state::Entity::EntityId entityId);
  void traceEntity(state::Entity::EntityId entityId);
  void pickupEntity(state::Entity::EntityId entityId);
};

} // namespace module

#endif // MODULE_SKILL_USE_MODULE_HPP_