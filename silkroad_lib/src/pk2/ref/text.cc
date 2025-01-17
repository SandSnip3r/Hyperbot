#include "pk2/ref/text.h"

namespace sro::pk2::ref {

std::ostream& operator<<(std::ostream &stream, const Text &text) {
	stream << "{service:" << (text.service ? "true":"false") << ','
         << "key: \"" << text.key << "\","
         << "english: \"" << text.english << "\"}";
	return stream;
}

} // namespace sro::pk2::ref