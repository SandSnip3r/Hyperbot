#ifndef RL_UI_TEXTURE_TO_QIMAGE_HPP_
#define RL_UI_TEXTURE_TO_QIMAGE_HPP_

#include <gli/texture2d.hpp>
#include <glm/common.hpp>
#include <gli/load_dds.hpp>
#include <gli/sampler2d.hpp>

#include <QImage>

#include <stdexcept>

namespace texture_to_image {

inline QImage rgbaDxt1UnormBlock8ToQImage(const gli::texture2d &texture2d) {
  if (texture2d.levels() < 1) {
    throw std::runtime_error("Have texture with no levels");
  }
  const auto &extent2d = texture2d.extent(0);
  QImage image(extent2d.x, extent2d.y, QImage::Format_RGBA8888);

  gli::extent2d blockExtent;
  {
    gli::extent3d tempExtent = gli::block_extent(gli::FORMAT_RGBA_DXT1_UNORM_BLOCK8);
    blockExtent.x = tempExtent.x;
    blockExtent.y = tempExtent.y;
  }

  gli::extent2d texelCoord;
  gli::extent2d blockCoord;
  gli::extent2d levelExtent = texture2d.extent(0);
  gli::extent2d levelExtentInBlocks = glm::max(gli::extent2d(1, 1), levelExtent / blockExtent);
  for (blockCoord.y = 0, texelCoord.y = 0; blockCoord.y < levelExtentInBlocks.y; ++blockCoord.y, texelCoord.y += blockExtent.y) {
    for (blockCoord.x = 0, texelCoord.x = 0; blockCoord.x < levelExtentInBlocks.x; ++blockCoord.x, texelCoord.x += blockExtent.x) {
      const gli::detail::dxt1_block *dxt1Block = texture2d.data<gli::detail::dxt1_block>(0, 0, 0) + (blockCoord.y * levelExtentInBlocks.x + blockCoord.x);
      const gli::detail::texel_block4x4 decompressedBlock = gli::detail::decompress_dxt1_block(*dxt1Block);

      gli::extent2d decompressedBlockCoord;
      for (decompressedBlockCoord.y = 0; decompressedBlockCoord.y < glm::min(4, levelExtent.y); ++decompressedBlockCoord.y) {
        for (decompressedBlockCoord.x = 0; decompressedBlockCoord.x < glm::min(4, levelExtent.x); ++decompressedBlockCoord.x) {
          const auto resultingCoordinate = texelCoord + decompressedBlockCoord;
          const auto &texel = decompressedBlock.Texel[decompressedBlockCoord.y][decompressedBlockCoord.x];
          image.setPixelColor(resultingCoordinate.x, resultingCoordinate.y, QColor::fromRgbF(texel.r, texel.g, texel.b, texel.a));
        }
      }
    }
  }
  return image;
}

inline QImage rgbaDxt3UnormBlock16ToQImage(const gli::texture2d &texture2d) {
  if (texture2d.levels() < 1) {
    throw std::runtime_error("Have texture with no levels");
  }
  const auto &extent2d = texture2d.extent(0);
  QImage image(extent2d.x, extent2d.y, QImage::Format_RGBA8888);

  gli::extent2d blockExtent;
  {
    gli::extent3d tempExtent = gli::block_extent(gli::FORMAT_RGBA_DXT3_UNORM_BLOCK16);
    blockExtent.x = tempExtent.x;
    blockExtent.y = tempExtent.y;
  }

  gli::extent2d texelCoord;
  gli::extent2d blockCoord;
  gli::extent2d levelExtent = texture2d.extent(0);
  gli::extent2d levelExtentInBlocks = glm::max(gli::extent2d(1, 1), levelExtent / blockExtent);
  for (blockCoord.y = 0, texelCoord.y = 0; blockCoord.y < levelExtentInBlocks.y; ++blockCoord.y, texelCoord.y += blockExtent.y) {
    for (blockCoord.x = 0, texelCoord.x = 0; blockCoord.x < levelExtentInBlocks.x; ++blockCoord.x, texelCoord.x += blockExtent.x) {
      const gli::detail::dxt3_block *dxt3Block = texture2d.data<gli::detail::dxt3_block>(0, 0, 0) + (blockCoord.y * levelExtentInBlocks.x + blockCoord.x);
      const gli::detail::texel_block4x4 decompressedBlock = gli::detail::decompress_dxt3_block(*dxt3Block);

      gli::extent2d decompressedBlockCoord;
      for (decompressedBlockCoord.y = 0; decompressedBlockCoord.y < glm::min(4, levelExtent.y); ++decompressedBlockCoord.y) {
        for (decompressedBlockCoord.x = 0; decompressedBlockCoord.x < glm::min(4, levelExtent.x); ++decompressedBlockCoord.x) {
          const auto resultingCoordinate = texelCoord + decompressedBlockCoord;
          const auto &texel = decompressedBlock.Texel[decompressedBlockCoord.y][decompressedBlockCoord.x];
          image.setPixelColor(resultingCoordinate.x, resultingCoordinate.y, QColor::fromRgbF(texel.r, texel.g, texel.b, texel.a));
        }
      }
    }
  }
  return image;
}

inline QImage bgra8UnormPack8ToQImage(const gli::texture2d &texture2d) {
  gli::texture2d::extent_type extent = texture2d.extent();
  QImage qtImage(extent.x, extent.y, QImage::Format_RGBA8888);
  for (gli::texture2d::size_type y = 0; y < extent.y; ++y) {
    for (gli::texture2d::size_type x = 0; x < extent.x; ++x) {
      gli::u8vec4 texel = texture2d.load<gli::u8vec4>(gli::texture2d::extent_type(x, y), 0);
      QRgb color = qRgba(texel.b, texel.g, texel.r, texel.a);
      qtImage.setPixel(x, y, color);
    }
  }
  return qtImage;
}

inline QImage bgr5a1UnormPack16ToQImage(const gli::texture2d &texture2d) {
  gli::texture2d::extent_type extent = texture2d.extent();
  QImage qtImage(extent.x, extent.y, QImage::Format_RGBA8888);
  for (gli::texture2d::size_type y = 0; y < extent.y; ++y) {
    for (gli::texture2d::size_type x = 0; x < extent.x; ++x) {
      gli::u16vec1 texel = texture2d.load<gli::u16vec1>(gli::texture2d::extent_type(x, y), 0);
      uint16_t packedValue = texel.x;
      uint8_t blue = (packedValue & 0x1F) << 3;
      uint8_t green = (packedValue >> 5 & 0x1F) << 3;
      uint8_t red = (packedValue >> 10 & 0x1F) << 3;
      uint8_t alpha = (packedValue >> 15) * 255;
      QRgb color = qRgba(red, green, blue, alpha);
      qtImage.setPixel(x, y, color);
    }
  }
  return qtImage;
}

inline QImage bgra4UnormPack16ToQImage(const gli::texture2d &texture2d) {
  gli::texture2d::extent_type extent = texture2d.extent();
  QImage qtImage(extent.x, extent.y, QImage::Format_RGBA8888);
  for (gli::texture2d::size_type y = 0; y < extent.y; ++y) {
    for (gli::texture2d::size_type x = 0; x < extent.x; ++x) {
      gli::u16vec1 texel = texture2d.load<gli::u16vec1>(gli::texture2d::extent_type(x, y), 0);
      uint16_t packedValue = texel.x;
      uint8_t blue = (packedValue & 0xF) << 4;
      uint8_t green = (packedValue >> 4 & 0xF) << 4;
      uint8_t red = (packedValue >> 8 & 0xF) << 4;
      uint8_t alpha = (packedValue >> 12) << 4;
      QRgb color = qRgba(red, green, blue, alpha);
      qtImage.setPixel(x, y, color);
    }
  }
  return qtImage;
}

inline QImage b5g6r5UnormPack16ToQImage(const gli::texture2d &texture2d) {
  gli::texture2d::extent_type extent = texture2d.extent();
  QImage qtImage(extent.x, extent.y, QImage::Format_RGB888);
  for (gli::texture2d::size_type y = 0; y < extent.y; ++y) {
    for (gli::texture2d::size_type x = 0; x < extent.x; ++x) {
      gli::u16vec1 texel = texture2d.load<gli::u16vec1>(gli::texture2d::extent_type(x, y), 0);
      uint16_t packedValue = texel.x;
      uint8_t blue = (packedValue & 0x1F) << 3;
      uint8_t green = (packedValue >> 5 & 0x3F) << 2;
      uint8_t red = (packedValue >> 11 & 0x1F) << 3;
      QRgb color = qRgb(red, green, blue);
      qtImage.setPixel(x, y, color);
    }
  }
  return qtImage;
}

inline QImage bgr8UnormPack32ToQImage(const gli::texture2d &texture2d) {
  gli::texture2d::extent_type extent = texture2d.extent();
  QImage qtImage(extent.x, extent.y, QImage::Format_RGB888);
  for (gli::texture2d::size_type y = 0; y < extent.y; ++y) {
    for (gli::texture2d::size_type x = 0; x < extent.x; ++x) {
      gli::u8vec4 texel = texture2d.load<gli::u8vec4>(gli::texture2d::extent_type(x, y), 0);
      QRgb color = qRgb(texel.b, texel.g, texel.r);
      qtImage.setPixel(x, y, color);
    }
  }
  return qtImage;
}

inline QImage texture2dToQImage(const gli::texture2d &texture2d) {
  switch (texture2d.format()) {
  case gli::FORMAT_RGBA_DXT1_UNORM_BLOCK8:
    return rgbaDxt1UnormBlock8ToQImage(texture2d);
  case gli::FORMAT_BGRA8_UNORM_PACK8:
    return bgra8UnormPack8ToQImage(texture2d);
  case gli::FORMAT_BGR5A1_UNORM_PACK16:
    return bgr5a1UnormPack16ToQImage(texture2d);
  case gli::FORMAT_B5G6R5_UNORM_PACK16:
    return b5g6r5UnormPack16ToQImage(texture2d);
  case gli::FORMAT_BGR8_UNORM_PACK32:
    return bgr8UnormPack32ToQImage(texture2d);
  case gli::FORMAT_RGBA_DXT3_UNORM_BLOCK16:
    return rgbaDxt3UnormBlock16ToQImage(texture2d);
  case gli::FORMAT_BGRA4_UNORM_PACK16:
    return bgra4UnormPack16ToQImage(texture2d);
  default:
    throw std::runtime_error("Unhandled .ddj format. Format is " + std::to_string(texture2d.format()));
  }
}

} // namespace texture_to_image

#endif // RL_UI_TEXTURE_TO_QIMAGE_HPP_

