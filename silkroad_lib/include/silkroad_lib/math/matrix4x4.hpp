#ifndef SRO_MATH_MATRIX_4X4_H_
#define SRO_MATH_MATRIX_4X4_H_

#include "vector3.hpp"

#include <array>
#include <ostream>

namespace sro::math {

struct Matrix4x4 {
public:
  std::array<std::array<float,4>,4> data;

  Matrix4x4();
  Matrix4x4(const std::initializer_list<std::initializer_list<float>> &init);

  void identity();

  void addTranslation(const Vector3 &v);

  void setTranslation(const Vector3 &v);
  void setRotation(float counterClockwiseAngle, const Vector3 &axis);

  Matrix4x4 inverse() const;

  Matrix4x4 operator*(const Matrix4x4 &m) const;
  Vector3 operator*(const Vector3 &v) const;
};

std::ostream& operator<<(std::ostream &stream, const Matrix4x4 &m);

} // namespace sro::math

#endif // SRO_MATH_MATRIX_4X4_H_