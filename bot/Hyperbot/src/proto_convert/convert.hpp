#ifndef PROTO_CONVERT_CONVERT_HPP_
#define PROTO_CONVERT_CONVERT_HPP_

#include "ui-proto/position.pb.h"

#include <silkroad_lib/position.h>

namespace proto_convert {

void positionToProto(const sro::Position &position, proto::position::Position &message);
sro::Position protoToPosition(const proto::position::Position &message);

} // namespace proto_convert

#endif // PROTO_CONVERT_CONVERT_HPP_