#ifndef PK2_DIVISION_INFO_HPP_
#define PK2_DIVISION_INFO_HPP_

#include "division.hpp"

#include <filesystem>
#include <ostream>
#include <vector>

namespace pk2 {

struct DivisionInfo {
	uint8_t locale; // ContentId
	std::vector<Division> divisions;
};

} // namespace pk2

std::ostream& operator<<(std::ostream &stream, const pk2::DivisionInfo &divisionInfo);

#endif // PK2_DIVISION_INFO_HPP_