#ifndef PACKET_BUILDING_COMMON_HPP_
#define PACKET_BUILDING_COMMON_HPP_

#include "storage/item.hpp"
#include "../../shared/stream_utility.h"

namespace packet::building {

void writeGenericItem(StreamUtility &stream, const storage::Item &item);

} // namespace packet::building

#endif // PACKET_BUILDING_COMMON_HPP_