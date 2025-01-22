#include "math/matrix4x4.hpp"

#include <cmath>

namespace sro::math {

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

void Matrix4x4::addTranslation(const Vector3 &v) {
  // Not sure why this is done instead of just adding to the third column
  Matrix4x4 m;
  m.setTranslation(v);
  *this = *this*m;
}

void Matrix4x4::setTranslation(const Vector3 &v) {
  data[0][3] += v.x;
  data[1][3] += v.y;
  data[2][3] += v.z;
}

void Matrix4x4::setRotation(float counterClockwiseAngle, const Vector3 &axis) {
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

Matrix4x4 Matrix4x4::inverse() const {
  const float A2323 = data[2][2] * data[3][3] - data[2][3] * data[3][2] ;
  const float A1323 = data[2][1] * data[3][3] - data[2][3] * data[3][1] ;
  const float A1223 = data[2][1] * data[3][2] - data[2][2] * data[3][1] ;
  const float A0323 = data[2][0] * data[3][3] - data[2][3] * data[3][0] ;
  const float A0223 = data[2][0] * data[3][2] - data[2][2] * data[3][0] ;
  const float A0123 = data[2][0] * data[3][1] - data[2][1] * data[3][0] ;
  const float A2313 = data[1][2] * data[3][3] - data[1][3] * data[3][2] ;
  const float A1313 = data[1][1] * data[3][3] - data[1][3] * data[3][1] ;
  const float A1213 = data[1][1] * data[3][2] - data[1][2] * data[3][1] ;
  const float A2312 = data[1][2] * data[2][3] - data[1][3] * data[2][2] ;
  const float A1312 = data[1][1] * data[2][3] - data[1][3] * data[2][1] ;
  const float A1212 = data[1][1] * data[2][2] - data[1][2] * data[2][1] ;
  const float A0313 = data[1][0] * data[3][3] - data[1][3] * data[3][0] ;
  const float A0213 = data[1][0] * data[3][2] - data[1][2] * data[3][0] ;
  const float A0312 = data[1][0] * data[2][3] - data[1][3] * data[2][0] ;
  const float A0212 = data[1][0] * data[2][2] - data[1][2] * data[2][0] ;
  const float A0113 = data[1][0] * data[3][1] - data[1][1] * data[3][0] ;
  const float A0112 = data[1][0] * data[2][1] - data[1][1] * data[2][0] ;

  const float det = 1 / (data[0][0] * ( data[1][1] * A2323 - data[1][2] * A1323 + data[1][3] * A1223 ) -
                         data[0][1] * ( data[1][0] * A2323 - data[1][2] * A0323 + data[1][3] * A0223 ) +
                         data[0][2] * ( data[1][0] * A1323 - data[1][1] * A0323 + data[1][3] * A0123 ) -
                         data[0][3] * ( data[1][0] * A1223 - data[1][1] * A0223 + data[1][2] * A0123 ) );

  return {
    {det *   ( data[1][1] * A2323 - data[1][2] * A1323 + data[1][3] * A1223 ),
     det * - ( data[0][1] * A2323 - data[0][2] * A1323 + data[0][3] * A1223 ),
     det *   ( data[0][1] * A2313 - data[0][2] * A1313 + data[0][3] * A1213 ),
     det * - ( data[0][1] * A2312 - data[0][2] * A1312 + data[0][3] * A1212 ),},
    {det * - ( data[1][0] * A2323 - data[1][2] * A0323 + data[1][3] * A0223 ),
     det *   ( data[0][0] * A2323 - data[0][2] * A0323 + data[0][3] * A0223 ),
     det * - ( data[0][0] * A2313 - data[0][2] * A0313 + data[0][3] * A0213 ),
     det *   ( data[0][0] * A2312 - data[0][2] * A0312 + data[0][3] * A0212 ),},
    {det *   ( data[1][0] * A1323 - data[1][1] * A0323 + data[1][3] * A0123 ),
     det * - ( data[0][0] * A1323 - data[0][1] * A0323 + data[0][3] * A0123 ),
     det *   ( data[0][0] * A1313 - data[0][1] * A0313 + data[0][3] * A0113 ),
     det * - ( data[0][0] * A1312 - data[0][1] * A0312 + data[0][3] * A0112 ),},
    {det * - ( data[1][0] * A1223 - data[1][1] * A0223 + data[1][2] * A0123 ),
     det *   ( data[0][0] * A1223 - data[0][1] * A0223 + data[0][2] * A0123 ),
     det * - ( data[0][0] * A1213 - data[0][1] * A0213 + data[0][2] * A0113 ),
     det *   ( data[0][0] * A1212 - data[0][1] * A0212 + data[0][2] * A0112 ),}
  };
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

Vector3 Matrix4x4::operator*(const Vector3 &v) const {
  Vector3 resultVector;
  resultVector.x = data[0][0]*v.x + data[0][1]*v.y + data[0][2]*v.z + data[0][3]*1;
  resultVector.y = data[1][0]*v.x + data[1][1]*v.y + data[1][2]*v.z + data[1][3]*1;
  resultVector.z = data[2][0]*v.x + data[2][1]*v.y + data[2][2]*v.z + data[2][3]*1;
  return resultVector;
}

std::ostream& operator<<(std::ostream &stream, const Matrix4x4 &m) {
  stream << '{' << m.data[0][0] << ',' << m.data[0][1] << ',' << m.data[0][2] << ',' << m.data[0][3] << "},";
  stream << '{' << m.data[1][0] << ',' << m.data[1][1] << ',' << m.data[1][2] << ',' << m.data[1][3] << "},";
  stream << '{' << m.data[2][0] << ',' << m.data[2][1] << ',' << m.data[2][2] << ',' << m.data[2][3] << "},";
  stream << '{' << m.data[3][0] << ',' << m.data[3][1] << ',' << m.data[3][2] << ',' << m.data[3][3] << '}';
  return stream;
}

} // namespace sro::math