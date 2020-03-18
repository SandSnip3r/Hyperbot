#include "characterInfo.hpp"

#include <vector>

namespace pk2::media {

std::ostream& operator<<(std::ostream &stream, const Character &character) {
	stream << "{id:" << character.id << ','
				 << "country:\"" << character.country << "\","
				 << "charGender:" << (int)character.charGender << '}';
	return stream;
}

} // namespace pk2::media