#ifndef ITEM_HPP
#define ITEM_HPP

#include <string>
#include <ostream>

namespace pk2 {

struct Item {
	int id;
	std::string codeName128;
	uint8_t typeId1;
	uint8_t typeId2;
	uint8_t typeId3;
	uint8_t typeId4;
};

std::ostream& operator<<(std::ostream &stream, const Item &item);

} // namespace pk2

#endif // ITEM_HPP