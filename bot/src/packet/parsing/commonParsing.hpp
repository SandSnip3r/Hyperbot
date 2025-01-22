#ifndef PACKET_PARSING_COMMON_HPP
#define PACKET_PARSING_COMMON_HPP

#include "entity/entity.hpp"
#include "packet/structures/packetInnerStructures.hpp"
#include "pk2/itemData.hpp"
#include "shared/stream_utility.h"
#include "storage/item.hpp"

#include <silkroad_lib/position.hpp>

#include <memory>

namespace packet::parsing {

std::shared_ptr<storage::Item> parseGenericItem(StreamUtility &stream, const pk2::ItemData &itemData);
structures::RentInfo parseRentInfo(StreamUtility &stream);
void parseItemCosSummoner(storage::ItemCosGrowthSummoner *cosSummoner, StreamUtility &stream);
void parseItem(storage::ItemEquipment &item, StreamUtility &stream);
void parseItem(storage::ItemCosGrowthSummoner &item, StreamUtility &stream);
void parseItem(storage::ItemCosAbilitySummoner &item, StreamUtility &stream);
void parseItem(storage::ItemMonsterCapsule &item, StreamUtility &stream);
void parseItem(storage::ItemStorage &item, StreamUtility &stream);
void parseItem(storage::ItemExpendable &item, StreamUtility &stream);
void parseItem(storage::ItemStone &item, StreamUtility &stream);
void parseItem(storage::ItemMagicPop &item, StreamUtility &stream);
void parseItem(storage::Item *item, StreamUtility &stream);
structures::SkillAction parseSkillAction(StreamUtility &stream);
sro::Position parsePosition(StreamUtility &stream);

// It would be nice to hold entities in a `std::unique_ptr`, however, that would require removing them from packets after parsing. When handling packets, the packets are usually passed as const&.
std::shared_ptr<entity::Entity> parseSpawn(StreamUtility &stream,
                                           const pk2::CharacterData &characterData,
                                           const pk2::ItemData &itemData,
                                           const pk2::SkillData &skillData,
                                           const pk2::TeleportData &teleportData);

} // namespace packet::parsing

#endif // PACKET_PARSING_COMMON_HPP