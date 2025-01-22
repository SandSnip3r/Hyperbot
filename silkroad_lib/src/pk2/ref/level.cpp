#include "pk2/ref/level.hpp"

namespace sro::pk2::ref {

std::ostream& operator<<(std::ostream &stream, const Level &level) {
	stream << "{lvl:" << static_cast<int>(level.lvl) << ','
         << "exp_C: \"" << level.exp_C << "\","
         << "exp_M: " << level.exp_M << '}';
	return stream;
}

} // namespace sro::pk2::ref