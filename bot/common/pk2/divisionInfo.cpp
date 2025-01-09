#include "divisionInfo.hpp"

std::ostream& operator<<(std::ostream &stream, const pk2::DivisionInfo &divisionInfo) {
	stream << "Locale: " << (int)divisionInfo.locale << ", divisions: {";
	for (int i=0; i<divisionInfo.divisions.size(); ++i) {
		stream << divisionInfo.divisions[i];
		if (i != divisionInfo.divisions.size()-1) {
			stream << ", ";
		}
	}
	stream << '}';
	return stream;
}