#ifndef DIVISION_HPP
#define DIVISION_HPP

#include <filesystem>
#include <ostream>
#include <vector>

namespace pk2 {

struct Division {
	std::string name;
	std::vector<std::string> gatewayIpAddresses;
};

} // namespace pk2

std::ostream& operator<<(std::ostream &stream, const pk2::Division &division);

#endif // DIVISION_HPP