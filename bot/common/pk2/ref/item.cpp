#include "item.hpp"

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

  std::vector<uint8_t> Item::elixirTargetItemTypeId3s() const {
    std::vector<uint8_t> result;
    // Supported item types are stored in param1, param5, and param6.
    for (const auto param : {param1, param5, param6}) {
      if (param != -1) {
        uint8_t a =  param        & 0xFF;
        uint8_t b = (param >>  8) & 0xFF;
        uint8_t c = (param >> 16) & 0xFF;
        uint8_t d = (param >> 24) & 0xFF;
        for (const auto val : {a, b, c, d}) {
          if (val != 0) {
            result.emplace_back(val);
          }
        }
      }
    }
    return result;
  }


} // namespace pk2::ref