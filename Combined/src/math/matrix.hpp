#ifndef MATRIX_HPP_
#define MATRIX_HPP_

#include "vector.hpp"

#include <array>
#include <iostream>
#include <ostream>

struct Matrix4x4 {
public:
  std::array<std::array<float,4>,4> data;

  Matrix4x4();
  Matrix4x4(const std::initializer_list<std::initializer_list<float>> &init);

  void identity();

  void addTranslation(const Vector &v);

  void setTranslation(const Vector &v);
  void setRotation(float counterClockwiseAngle, const Vector &axis);

  Matrix4x4 operator*(const Matrix4x4 &m) const;
  Vector operator*(const Vector &v) const;
};

std::ostream& operator<<(std::ostream &stream, const Matrix4x4 &m);

#endif // MATRIX_HPP_