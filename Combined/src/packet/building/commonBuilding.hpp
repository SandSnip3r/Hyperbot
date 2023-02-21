#ifndef PACKET_BUILDING_COMMON_HPP_
#define PACKET_BUILDING_COMMON_HPP_

#include "packet/structures/packetInnerStructures.hpp"
#include "storage/item.hpp"
#include "../../shared/stream_utility.h"

#include <silkroad_lib/position.h>

namespace packet::building {

void writeGenericItem(StreamUtility &stream, const storage::Item &item);
void writePosition(StreamUtility &stream, const sro::Position &position);

// This function was written for a hack and may not have full functionality
// TODO: Finish
void writeSkillAction(StreamUtility &stream, const structures::SkillAction &action);

// This function was written for a hack and may not have full functionality
// TODO: Finish
void writeHitObject(StreamUtility &stream, const structures::SkillActionHitObject &hitObject);

// This function was written for a hack and may not have full functionality
// TODO: Finish
void writeHit(StreamUtility &stream, const structures::SkillActionHitResult &hit);

} // namespace packet::building

#endif // PACKET_BUILDING_COMMON_HPP_