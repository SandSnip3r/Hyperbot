#ifndef PK2_DIVISION_INFO_HPP_
#define PK2_DIVISION_INFO_HPP_

#include <cstdint>
#include <string>
#include <vector>

namespace sro::pk2 {

struct Division {
	std::string name;
	std::vector<std::string> gatewayIpAddresses;
  std::string toString() const;
};

struct DivisionInfo {
	uint8_t locale; // ContentId
	std::vector<Division> divisions;
  std::string toString() const;
};

} // namespace sro::pk2


#endif // PK2_DIVISION_INFO_HPP_