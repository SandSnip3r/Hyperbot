#ifndef PACKET_BUILDING_FRAMEWORK_ALIVE_NOTIFY_HPP_
#define PACKET_BUILDING_FRAMEWORK_ALIVE_NOTIFY_HPP_

#include "packet/opcode.hpp"
#include "shared/silkroad_security.h"

namespace packet::building {

class FrameworkAliveNotify {
private:
  static const Opcode kOpcode_ = Opcode::kFrameworkAliveNotify;
  static const bool kEncrypted_ = false;
  static const bool kMassive_ = false;
public:
  static PacketContainer packet();
};

} // namespace packet::building

#endif // PACKET_BUILDING_FRAMEWORK_ALIVE_NOTIFY_HPP_