#ifndef COMMON_DYNAMIC_UNIQUE_CAST_HPP_
#define COMMON_DYNAMIC_UNIQUE_CAST_HPP_

#include <memory>

namespace common {

template <typename DerivedType, typename BaseType>
std::unique_ptr<DerivedType> dynamicUniqueCast(std::unique_ptr<BaseType>&& basePtr) {
  if (auto derivedPtr = dynamic_cast<DerivedType*>(basePtr.get())) {
    basePtr.release();  // Release ownership so we can transfer it
    return std::unique_ptr<DerivedType>(derivedPtr);
  }
  return nullptr;
}

} // namespace common

#endif // COMMON_DYNAMIC_UNIQUE_CAST_HPP_