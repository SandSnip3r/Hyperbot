#include "opcode.hpp"

#include <absl/log/log.h>
#include <absl/strings/str_format.h>

namespace packet {

std::string_view toString(Opcode opcode) {
#define F(name, value) if (opcode == Opcode::k##name) { return #name; }
  PACKET_OPCODE_LIST(F)
#undef F
  VLOG(1) << absl::StreamFormat("Asking for string of unknown Opcode: 0x%X", static_cast<int>(opcode));
  return "UNKNOWN";
}

} // namespace packet