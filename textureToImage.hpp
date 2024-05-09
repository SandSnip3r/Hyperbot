#ifndef TEXTURETOIMAGE_HPP_
#define TEXTURETOIMAGE_HPP_

#include <QImage>

namespace gli {
class texture2d;
}

namespace texture_to_image {
QImage dataToQImage(char const * Data, std::size_t Size);
} // namespace texture_to_image

#endif // TEXTURETOIMAGE_HPP_