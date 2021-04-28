#include "matrix.hpp"

#include <cmath>

Matrix4x4::Matrix4x4() {
  identity();
}

Matrix4x4::Matrix4x4(const std::initializer_list<std::initializer_list<float>> &init) {
  auto dataIt = data.begin();
  for (const auto &row : init) {
    std::copy(row.begin(), row.end(), dataIt->begin());
    ++dataIt;
  }
}

void Matrix4x4::identity() {
  data[0] = {1,0,0,0};
  data[1] = {0,1,0,0};
  data[2] = {0,0,1,0};
  data[3] = {0,0,0,1};
}

void Matrix4x4::addTranslation(const Vector &v) {
  // Not sure why this is done instead of just adding to the third column
  Matrix4x4 m;
  m.setTranslation(v);
  *this = *this*m;
}

void Matrix4x4::setTranslation(const Vector &v) {
  data[0][3] += v.x;
  data[1][3] += v.y;
  data[2][3] += v.z;
}

void Matrix4x4::setRotation(float counterClockwiseAngle, const Vector &axis) {
  if (axis.length()-1 > 0.0000001) {
    throw std::runtime_error("Matrix4x4::setRotation: Axis vector should be normalized");
  }

  float c = cos(counterClockwiseAngle);
  float s = sin(counterClockwiseAngle);
  float t = 1-c;

  data[0][0] = axis.x*axis.x*t + c;
  data[0][1] = axis.x*axis.y*t - axis.z*s;
  data[0][2] = axis.x*axis.z*t + axis.y*s;

  data[1][0] = axis.y*axis.x*t + axis.z*s;
  data[1][1] = axis.y*axis.y*t + c;
  data[1][2] = axis.y*axis.z*t - axis.x*s;

  data[2][0] = axis.z*axis.x*t - axis.y*s;
  data[2][1] = axis.z*axis.y*t + axis.x*s;
  data[2][2] = axis.z*axis.z*t + c;
}

Matrix4x4 Matrix4x4::operator*(const Matrix4x4 &m) const {
  Matrix4x4 result;
  result.data[0][0] = data[0][0]*m.data[0][0] + data[0][1]*m.data[1][0] + data[0][2]*m.data[2][0] + data[0][3]*m.data[3][0];
  result.data[0][1] = data[0][0]*m.data[0][1] + data[0][1]*m.data[1][1] + data[0][2]*m.data[2][1] + data[0][3]*m.data[3][1];
  result.data[0][2] = data[0][0]*m.data[0][2] + data[0][1]*m.data[1][2] + data[0][2]*m.data[2][2] + data[0][3]*m.data[3][2];
  result.data[0][3] = data[0][0]*m.data[0][3] + data[0][1]*m.data[1][3] + data[0][2]*m.data[2][3] + data[0][3]*m.data[3][3];
  result.data[1][0] = data[1][0]*m.data[0][0] + data[1][1]*m.data[1][0] + data[1][2]*m.data[2][0] + data[1][3]*m.data[3][0];
  result.data[1][1] = data[1][0]*m.data[0][1] + data[1][1]*m.data[1][1] + data[1][2]*m.data[2][1] + data[1][3]*m.data[3][1];
  result.data[1][2] = data[1][0]*m.data[0][2] + data[1][1]*m.data[1][2] + data[1][2]*m.data[2][2] + data[1][3]*m.data[3][2];
  result.data[1][3] = data[1][0]*m.data[0][3] + data[1][1]*m.data[1][3] + data[1][2]*m.data[2][3] + data[1][3]*m.data[3][3];
  result.data[2][0] = data[2][0]*m.data[0][0] + data[2][1]*m.data[1][0] + data[2][2]*m.data[2][0] + data[2][3]*m.data[3][0];
  result.data[2][1] = data[2][0]*m.data[0][1] + data[2][1]*m.data[1][1] + data[2][2]*m.data[2][1] + data[2][3]*m.data[3][1];
  result.data[2][2] = data[2][0]*m.data[0][2] + data[2][1]*m.data[1][2] + data[2][2]*m.data[2][2] + data[2][3]*m.data[3][2];
  result.data[2][3] = data[2][0]*m.data[0][3] + data[2][1]*m.data[1][3] + data[2][2]*m.data[2][3] + data[2][3]*m.data[3][3];
  result.data[3][0] = data[3][0]*m.data[0][0] + data[3][1]*m.data[1][0] + data[3][2]*m.data[2][0] + data[3][3]*m.data[3][0];
  result.data[3][1] = data[3][0]*m.data[0][1] + data[3][1]*m.data[1][1] + data[3][2]*m.data[2][1] + data[3][3]*m.data[3][1];
  result.data[3][2] = data[3][0]*m.data[0][2] + data[3][1]*m.data[1][2] + data[3][2]*m.data[2][2] + data[3][3]*m.data[3][2];
  result.data[3][3] = data[3][0]*m.data[0][3] + data[3][1]*m.data[1][3] + data[3][2]*m.data[2][3] + data[3][3]*m.data[3][3];
  return result;
}

Vector Matrix4x4::operator*(const Vector &v) const {
  Vector resultVector;
  resultVector.x = data[0][0]*v.x + data[0][1]*v.y + data[0][2]*v.z + data[0][3]*1;
  resultVector.y = data[1][0]*v.x + data[1][1]*v.y + data[1][2]*v.z + data[1][3]*1;
  resultVector.z = data[2][0]*v.x + data[2][1]*v.y + data[2][2]*v.z + data[2][3]*1;
  return resultVector;
}


//==========================================================================================

std::ostream& operator<<(std::ostream &stream, const Matrix4x4 &m) {
  stream << '{' << m.data[0][0] << ',' << m.data[0][1] << ',' << m.data[0][2] << ',' << m.data[0][3] << "},";
  stream << '{' << m.data[1][0] << ',' << m.data[1][1] << ',' << m.data[1][2] << ',' << m.data[1][3] << "},";
  stream << '{' << m.data[2][0] << ',' << m.data[2][1] << ',' << m.data[2][2] << ',' << m.data[2][3] << "},";
  stream << '{' << m.data[3][0] << ',' << m.data[3][1] << ',' << m.data[3][2] << ',' << m.data[3][3] << '}';
  return stream;
}