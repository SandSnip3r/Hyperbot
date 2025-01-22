#ifndef PACKET_BUILDING_COMMON_HPP_
#define PACKET_BUILDING_COMMON_HPP_

#include "packet/structures/packetInnerStructures.hpp"
#include "shared/stream_utility.h"
#include "storage/item.hpp"

#include <silkroad_lib/position.hpp>

namespace packet::building {

struct NetworkReadyPosition {
public:
  NetworkReadyPosition(const sro::Position &pos);
  static NetworkReadyPosition roundToNearest(const sro::Position &pos);
  sro::Position asSroPosition() const;
  static sro::Position truncateForNetwork(const sro::Position &pos);
private:
  sro::Position convertedPosition_;
};

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