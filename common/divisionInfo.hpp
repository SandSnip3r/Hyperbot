#ifndef DIVISION_INFO_HPP
#define DIVISION_INFO_HPP

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

#endif // DIVISION_INFO_HPP