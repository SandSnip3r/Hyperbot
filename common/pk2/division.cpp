#include "division.hpp"

std::ostream& operator<<(std::ostream &stream, const pk2::Division &division) {
	stream << division.name << ": [";
	for (int i=0; i<division.gatewayIpAddresses.size(); ++i) {
		stream << division.gatewayIpAddresses[i];
		if (i != division.gatewayIpAddresses.size()-1) {
			stream << ", ";
		}
	}
	stream << ']';
	return stream;
}