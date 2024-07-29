#ifndef PACKET_PARSING_FRAMEWORK_MESSAGE_IDENTIFY_HPP_
#define PACKET_PARSING_FRAMEWORK_MESSAGE_IDENTIFY_HPP_

#include "packet/parsing/parsedPacket.hpp"

#include <string>
#include <string_view>

namespace packet::parsing {

class FrameworkMessageIdentify : public ParsedPacket {
public:
  FrameworkMessageIdentify(const PacketContainer &packet);
  std::string_view moduleName() const { return moduleName_; }
  uint8_t isCertified() const { return isCertified_; }
private:
  std::string moduleName_;
  uint8_t isCertified_;
};

} // namespace packet::parsing

#endif // PACKET_PARSING_FRAMEWORK_MESSAGE_IDENTIFY_HPP_