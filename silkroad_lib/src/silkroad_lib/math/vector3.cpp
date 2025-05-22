#include <silkroad_lib/math/vector3.hpp>

#include <cmath>

namespace sro::math {

Vector3::Vector3(float xParam, float yParam, float zParam) : x(xParam), y(yParam), z(zParam) {

}

Vector3& Vector3::operator=(const Vector3 &other) {
  x = other.x;
  y = other.y;
  z = other.z;
  return *this;
}

float Vector3::length() const {
  return sqrt(x*x+y*y+z*z);
}

//==================================================================================================

bool operator==(const Vector3 &v1, const Vector3 &v2) {
  return (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z);
}

bool operator!=(const Vector3 &v1, const Vector3 &v2) {
  return !(v1 == v2);
}

std::ostream& operator<<(std::ostream &stream, const Vector3 &v) {
  stream << v.x << ',' << v.y << ',' << v.z;
  return stream;
}

} // namespace sro::math