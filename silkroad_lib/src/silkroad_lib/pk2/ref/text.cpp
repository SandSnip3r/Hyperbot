#include <silkroad_lib/pk2/ref/text.hpp>

namespace sro::pk2::ref {

std::ostream& operator<<(std::ostream &stream, const Text &text) {
	stream << "{service:" << (text.service ? "true":"false") << ','
         << "codeName128: \"" << text.codeName128 << "\","
         << "english: \"" << text.english << "\"}";
	return stream;
}

} // namespace sro::pk2::ref