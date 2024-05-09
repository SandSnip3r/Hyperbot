#include "convert.hpp"

namespace proto_convert {

void positionToProto(const sro::Position &position, proto::position::Position &message) {
  message.set_regionid(position.regionId());
  message.set_x(position.xOffset());
  message.set_y(position.yOffset());
  message.set_z(position.zOffset());
}

sro::Position protoToPosition(const proto::position::Position &message) {
  return sro::Position(message.regionid(), message.x(), message.y(), message.z());
}

} // namespace proto_convert