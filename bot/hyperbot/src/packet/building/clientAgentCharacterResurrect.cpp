#include "clientAgentCharacterResurrect.hpp"

#include <absl/log/log.h>
#include <absl/strings/str_format.h>

#include <stdexcept>

namespace packet::building {

PacketContainer ClientAgentCharacterResurrect::resurrect(packet::enums::ResurrectionOptionFlag option) {
  StreamUtility stream;
  // 1   byte    resurrectType // 1 = Regular, 2 = Beginner
  // Regular = Resurrect at the specified point. / Resurrect at the present point.
  if (option == packet::enums::ResurrectionOptionFlag::kNormal) {
    LOG(INFO) << "Want to resurrect with option " << static_cast<int>(option) << ". Sending kAtSpecifiedPoint";
    stream.Write(packet::enums::ResurrectionOptionFlag::kAtSpecifiedPoint);
  } else {
    throw std::runtime_error(absl::StrFormat("Invalid resurrection option %d", static_cast<int>(option)));
  }
  return PacketContainer(static_cast<uint16_t>(kOpcode_), stream, (kEncrypted_ ? 1 : 0), (kMassive_ ? 1 : 0));
}

} // namespace packet::building