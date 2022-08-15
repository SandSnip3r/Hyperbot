#include "magicOption.hpp"

namespace pk2::ref {

std::ostream& operator<<(std::ostream &stream, const MagicOption &magOpt) {
	stream << "{id:" << magOpt.id << ','
         << "mOptName128: \"" << magOpt.mOptName128 << "\","
         << "attrType: " << magOpt.attrType << ','
         << "mLevel: " << magOpt.mLevel << '}';
	return stream;
}

} // namespace pk2::ref