#ifndef PACKET_PARSING_COMMON_HPP
#define PACKET_PARSING_COMMON_HPP

#include "../structures/packetInnerStructures.hpp"
#include "../../storage/item.hpp"
#include "../../shared/stream_utility.h"

namespace packet::parsing {

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

} // namespace packet::parsing

#endif // PACKET_PARSING_COMMON_HPP