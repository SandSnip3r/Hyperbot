#include "pk2/ref/magicOption.hpp"

namespace sro::pk2::ref {

std::ostream& operator<<(std::ostream &stream, const MagicOption &magOpt) {
	stream << "{id:" << magOpt.id << ','
         << "mOptName128: \"" << magOpt.mOptName128 << "\","
         << "attrType: " << magOpt.attrType << ','
         << "mLevel: " << magOpt.mLevel << '}';
	return stream;
}

} // namespace sro::pk2::ref