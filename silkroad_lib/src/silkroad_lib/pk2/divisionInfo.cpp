#include <silkroad_lib/pk2/divisionInfo.hpp>

#include <sstream>

namespace sro::pk2 {

std::string Division::toString() const {
  std::stringstream stream;
	stream << name << ": [";
	for (int i=0; i<gatewayIpAddresses.size(); ++i) {
		stream << gatewayIpAddresses[i];
		if (i != gatewayIpAddresses.size()-1) {
			stream << ", ";
		}
	}
	stream << ']';
	return stream.str();
}

std::string DivisionInfo::toString() const {
  std::stringstream stream;
	stream << "Locale: " << (int)locale << ", divisions: {";
	for (int i=0; i<divisions.size(); ++i) {
		stream << divisions[i].toString();
		if (i != divisions.size()-1) {
			stream << ", ";
		}
	}
	stream << '}';
	return stream.str();
}

} // namespace sro::pk2