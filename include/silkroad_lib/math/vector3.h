#ifndef SRO_MATH_VECTOR_3_H_
#define SRO_MATH_VECTOR_3_H_

#include <ostream>

namespace sro::math {

struct Vector3 {
public:
  float x, y{0.0f}, z;
  Vector3() = default;
  Vector3(float xParam, float yParam, float zParam);
  Vector3& operator=(const Vector3 &other);

  float length() const;
};

bool operator==(const Vector3 &v1, const Vector3 &v2);
bool operator!=(const Vector3 &v1, const Vector3 &v2);
std::ostream& operator<<(std::ostream &stream, const Vector3 &v);

} // namespace sro::math

#endif // SRO_MATH_VECTOR_3_H_