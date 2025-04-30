#include "event/eventCode.hpp"

#include <absl/strings/str_format.h>

#include <stdexcept>

namespace event {

std::string_view toString(EventCode eventCode) {
  #define F(name) if (eventCode == EventCode::k##name) { return #name; }
    EVENT_EVENTCODE_LIST(F)
  #undef F
    throw std::runtime_error(absl::StrFormat("Unknown EventCode: %d", static_cast<int>(eventCode)));
  }

} // namespace event