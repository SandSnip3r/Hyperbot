#ifndef PACKET_PARSING_COMMON_HPP
#define PACKET_PARSING_COMMON_HPP

#include "packet/structures/packetInnerStructures.hpp"
#include "pk2/itemData.hpp"
#include "storage/item.hpp"
#include "../../shared/stream_utility.h"

#include <silkroad_lib/position.h>

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

} // namespace packet::parsing

#endif // PACKET_PARSING_COMMON_HPP