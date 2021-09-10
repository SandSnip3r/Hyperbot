#ifndef VECTOR_HPP_
#define VECTOR_HPP_

#include <array>
#include <ostream>

namespace math {

struct Vector {
public:
  float x,y,z;
  Vector() = default;
  Vector(float xParam, float yParam, float zParam);
  Vector& operator=(const Vector &other);

  float length() const;
};

bool operator==(const Vector &v1, const Vector &v2);
bool operator!=(const Vector &v1, const Vector &v2);
std::ostream& operator<<(std::ostream &stream, const Vector &v);

} // namespace math

#endif // VECTOR_HPP_