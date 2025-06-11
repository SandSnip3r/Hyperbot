#ifndef HELPERS_HPP_
#define HELPERS_HPP_

#include "packet/enums/packetEnums.hpp"
#include "packet/structures/packetInnerStructures.hpp"
#include "storage/item.hpp"
#include "storage/storage.hpp"

#include <silkroad_lib/pk2/gameData.hpp>
#include <silkroad_lib/pk2/ref/item.hpp>
#include <silkroad_lib/pk2/ref/scrapOfPackageItem.hpp>
#include <silkroad_lib/position.hpp>

#include <filesystem>
#include <map>
#include <memory>
#include <tuple>

namespace helpers {

std::filesystem::path getAppDataDirectory();
float secondsToTravel(const sro::Position &srcPosition, const sro::Position &destPosition, const float currentSpeed);
void initializeInventory(storage::Storage &inventory, uint8_t inventorySize, const std::map<uint8_t, std::shared_ptr<storage::Item>> &inventoryItemMap);
void printItem(uint8_t slot, const storage::Item *item, const sro::pk2::GameData &gameData);

template <auto F>
constexpr int toBitNum() {
  // F must be an enum
  using E = decltype(F);
  static_assert(std::is_enum<E>::value, "toBitNum<F>: F must be an enum value");

  // Underlying type must be one of the allowed unsigned widths
  using U = std::underlying_type_t<E>;
  static_assert(std::is_unsigned<U>::value, "toBitNum<F>: underlying type must be unsigned");
  static_assert(sizeof(U) == 1 || sizeof(U) == 2 || sizeof(U) == 4 || sizeof(U) == 8, "toBitNum<F>: underlying type must be 8, 16, 32 or 64 bits");

  // Pull out the raw bits
  constexpr U num = static_cast<U>(F);

  // Exactly one bit must be set
  static_assert(num != 0, "toBitNum<F>: no bit is set");
  static_assert((num & (num - 1)) == 0, "toBitNum<F>: multiple bits are set");

  // Find and return the zero‐based index of that bit
  for (int i = 0; i < static_cast<int>(sizeof(U) * 8); ++i) {
    if (num & (U(1) << i)) {
      return i;
    }
  }
  return -1; // unreachable, but suppresses compiler warnings
}

template <class E>
constexpr int toBitNum(E stateFlag) {
  static_assert(std::is_enum<E>::value, "toBitNum(E): E must be an enum");
  using U = std::underlying_type_t<E>;
  static_assert(std::is_unsigned<U>::value, "toBitNum(E): underlying type must be unsigned");
  static_assert(sizeof(U)==1||sizeof(U)==2||sizeof(U)==4||sizeof(U)==8, "toBitNum(E): underlying size must be 8/16/32/64 bits");

  U num = static_cast<U>(stateFlag);
  if (num == 0) {
    throw std::runtime_error("toBitNum: no bit set");
  }
  if ((num & (num - 1)) != 0) {
    throw std::runtime_error("toBitNum: multiple bits set");
  }

  for (int i = 0; i < int(sizeof(U)*8); ++i) {
    if (num & (U(1)<<i)) {
      return i;
    }
  }

  // unreachable
  throw std::runtime_error("toBitNum: logic error");
}

template <typename E, int N>
constexpr E fromBitNum() {
  static_assert(std::is_enum<E>::value, "fromBitNum<E,N>: E must be an enum type");
  using U = std::underlying_type_t<E>;
  static_assert(std::is_unsigned<U>::value, "fromBitNum<E,N>: underlying type must be unsigned");
  static_assert(sizeof(U)==1||sizeof(U)==2||sizeof(U)==4||sizeof(U)==8, "fromBitNum<E,N>: underlying size must be 8/16/32/64 bits");
  static_assert(N >= 0 && N < static_cast<int>(sizeof(U)*8), "fromBitNum<E,N>: bit index out of range");
  return static_cast<E>( U(1) << N );
}

template <typename E>
constexpr E fromBitNum(int bitIndex) {
  static_assert(std::is_enum<E>::value, "fromBitNum<E>: E must be an enum type");
  using U = std::underlying_type_t<E>;
  static_assert(std::is_unsigned<U>::value, "fromBitNum<E>: underlying type must be unsigned");
  static_assert(sizeof(U)==1||sizeof(U)==2||sizeof(U)==4||sizeof(U)==8, "fromBitNum<E>: underlying size must be 8/16/32/64 bits");

  if (bitIndex < 0 || bitIndex >= static_cast<int>(sizeof(U)*8)) {
    throw std::out_of_range("fromBitNum: bit index out of range");
  }
  return static_cast<E>( U(1) << bitIndex );
}

std::shared_ptr<storage::Item> createItemFromScrap(const sro::pk2::ref::ScrapOfPackageItem &itemScrap, const sro::pk2::ref::Item &itemRef);

namespace type_id {

// TODO: Move uses of this type Id stuff to use the new type id categories system
std::tuple<uint8_t,uint8_t,uint8_t,uint8_t> splitTypeId(const uint16_t typeId);

} // namespace type_id

} // namespace helpers

#endif // HELPERS_HPP_
