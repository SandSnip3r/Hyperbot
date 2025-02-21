#include "intelligenceActor.hpp"

#include "bot.hpp"
#include "packet/building/clientAgentInventoryItemUseRequest.hpp"
#include "packet/building/clientAgentActionCommandRequest.hpp"
#include "type_id/categories.hpp"

#include <absl/log/log.h>

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

} // namespace

namespace state::machine {

IntelligenceActor::IntelligenceActor(Bot &bot, sro::scalar_types::EntityGlobalId opponentGlobalId) : StateMachine(bot), opponentGlobalId_(opponentGlobalId) {
  LOG(INFO) << "Instantiated intelligence actor!";

}

IntelligenceActor::~IntelligenceActor() {
}

Status IntelligenceActor::onUpdate(const event::Event *event) {
  static std::mt19937 randomEngine = createRandomEngine();

  std::discrete_distribution actionTypeDist({
    4, // Attack #1
    4, // Attack #2
    4, // Attack #3
    4, // Attack #4
    2, // Imbue
    2, // Use HP potion
    1, // Use MP potion
    60, // Do nothing
  });

  // Blood Blade Force:610}, {Flower Bloom Blade:644

  const int val = actionTypeDist(randomEngine);
  if (val == 0) {
    // Cast an attack on our target.
    constexpr sro::scalar_types::ReferenceSkillId kBloodChain{339};
    VLOG(1) << "Sending packet to cast";
    bot_.packetBroker().injectPacket(packet::building::ClientAgentActionCommandRequest::cast(kBloodChain, opponentGlobalId_), PacketContainer::Direction::kClientToServer);
  } else if (val == 1) {
    // Cast an attack on our target.
    constexpr sro::scalar_types::ReferenceSkillId kBillowChain{371};
    VLOG(1) << "Sending packet to cast";
    bot_.packetBroker().injectPacket(packet::building::ClientAgentActionCommandRequest::cast(kBillowChain, opponentGlobalId_), PacketContainer::Direction::kClientToServer);
  } else if (val == 2) {
    // Cast an attack on our target.
    constexpr sro::scalar_types::ReferenceSkillId kBloodBladeForce{610};
    VLOG(1) << "Sending packet to cast";
    bot_.packetBroker().injectPacket(packet::building::ClientAgentActionCommandRequest::cast(kBloodBladeForce, opponentGlobalId_), PacketContainer::Direction::kClientToServer);
  } else if (val == 3) {
    // Cast an attack on our target.
    constexpr sro::scalar_types::ReferenceSkillId kFlowerBloomBlade{644};
    VLOG(1) << "Sending packet to cast";
    bot_.packetBroker().injectPacket(packet::building::ClientAgentActionCommandRequest::cast(kFlowerBloomBlade, opponentGlobalId_), PacketContainer::Direction::kClientToServer);
  } else if (val == 4) {
    // Cast an attack on our target.
    constexpr sro::scalar_types::ReferenceSkillId kExtremeFireForce{1380};
    VLOG(1) << "Sending packet to cast";
    bot_.packetBroker().injectPacket(packet::building::ClientAgentActionCommandRequest::cast(kExtremeFireForce), PacketContainer::Direction::kClientToServer);
  } else if (val == 5) {
    // Lets use a health potion.
    const sro::pk2::ref::ItemId smallHpPotionRefId = bot_.gameData().itemData().getItemId([](const sro::pk2::ref::Item &item) {
      return type_id::categories::kHpPotion.contains(type_id::getTypeId(item)) && item.itemClass == 2;
    });
    useItem(smallHpPotionRefId);
  } else if (val == 6) {
    // Lets use a mana potion.
    const sro::pk2::ref::ItemId smallMpPotionRefId = bot_.gameData().itemData().getItemId([](const sro::pk2::ref::Item &item) {
      return type_id::categories::kMpPotion.contains(type_id::getTypeId(item)) && item.itemClass == 2;
    });
    useItem(smallMpPotionRefId);
  } else if (val == 7) {
    // Skip this update.
  }
  return Status::kNotDone;
}

void IntelligenceActor::useItem(sro::pk2::ref::ItemId refId) {
  // Find this item in our inventory.
  for (int slot=0; slot<bot_.inventory().size(); ++slot) {
    if (!bot_.inventory().hasItem(slot)) {
      // No item here.
      continue;
    }
    const storage::Item *item = bot_.inventory().getItem(slot);
    if (item->refItemId == refId) {
      // Use this item.
      VLOG(1) << "Sending packet to use item at slot " << slot;
      bot_.packetBroker().injectPacket(packet::building::ClientAgentInventoryItemUseRequest::packet(slot, item->typeId()), PacketContainer::Direction::kClientToServer);
      break;
    }
  }
}

} // namespace state::machine
