#include "item.hpp"

#include <vector>

namespace pk2::ref {

std::ostream& operator<<(std::ostream &stream, const Item &item) {
	stream << "{id:" << item.id << ','
				 << "codeName128:\"" << item.codeName128 << "\","
				 << "cashItem:" << (int)item.cashItem << ','
				 << "bionic:" << (int)item.bionic << ','
				 << "typeId1:" << (int)item.typeId1 << ','
				 << "typeId2:" << (int)item.typeId2 << ','
				 << "typeId3:" << (int)item.typeId3 << ','
				 << "typeId4:" << (int)item.typeId4 << '}';
	return stream;
}

} // namespace pk2::ref