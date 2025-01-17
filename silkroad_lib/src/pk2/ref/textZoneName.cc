#include "pk2/ref/textZoneName.h"

namespace sro::pk2::ref {

std::ostream& operator<<(std::ostream &stream, const TextZoneName &zoneName) {
	stream << "{service:" << (zoneName.service ? "true":"false") << ','
         << "key: \"" << zoneName.key << "\","
         << "english: \"" << zoneName.english << "\"}";
	return stream;
}

} // namespace sro::pk2::ref