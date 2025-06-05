#include <silkroad_lib/pk2/ref/textZoneName.hpp>

namespace sro::pk2::ref {

std::ostream& operator<<(std::ostream &stream, const TextZoneName &zoneName) {
	stream << "{service:" << (zoneName.service ? "true":"false") << ','
         << "codeName128: \"" << zoneName.codeName128 << "\","
         << "english: \"" << zoneName.english << "\"}";
	return stream;
}

} // namespace sro::pk2::ref