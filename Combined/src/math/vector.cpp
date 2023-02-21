#include "vector.hpp"

#include <cmath>

namespace math {

Vector::Vector(float xParam, float yParam, float zParam) : x(xParam), y(yParam), z(zParam) {

}

Vector& Vector::operator=(const Vector &other) {
  x = other.x;
  y = other.y;
  z = other.z;
  return *this;
}

float Vector::length() const {
  return sqrt(x*x+y*y+z*z);
}

//==================================================================================================

Vector operator-(const Vector &v) {
  return {-v.x, -v.y, -v.z};
}

bool operator==(const Vector &v1, const Vector &v2) {
  return (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z);
}

bool operator!=(const Vector &v1, const Vector &v2) {
  return !(v1 == v2);
}

std::ostream& operator<<(std::ostream &stream, const Vector &v) {
  stream << v.x << ',' << v.y << ',' << v.z;
  return stream;
}

} // namespace math