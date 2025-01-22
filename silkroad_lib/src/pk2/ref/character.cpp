#include "pk2/ref/character.hpp"

#include <vector>

namespace sro::pk2::ref {

std::ostream& operator<<(std::ostream &stream, const Character &character) {
	stream << "{id:" << character.id << ','
				 << "codeName128:\"" << character.codeName128 << "\","
				 << "typeId1:\"" << (int)character.typeId1 << "\","
				 << "typeId2:\"" << (int)character.typeId2 << "\","
				 << "typeId3:\"" << (int)character.typeId3 << "\","
				 << "typeId4:\"" << (int)character.typeId4 << "\","
				 << "country:\"" << character.country << "\","
				 << "charGender:" << (int)character.charGender << '}';
	return stream;
}

} // namespace sro::pk2::ref