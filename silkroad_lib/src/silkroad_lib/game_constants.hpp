#ifndef SRO_GAME_CONSTANTS_H_
#define SRO_GAME_CONSTANTS_H_

#include <silkroad_lib/scalar_types.hpp>

#include <chrono>

namespace sro::game_constants {

constexpr float kRegionWidth{1920.0f};
constexpr float kRegionHeight{kRegionWidth};
constexpr float kRegionSize{kRegionWidth};

constexpr scalar_types::StorageIndexType kFirstInventorySlot{13};

constexpr scalar_types::StorageIndexType kAvatarHatSlot{1};
constexpr scalar_types::StorageIndexType kAvatarDressSlot{2};
constexpr scalar_types::StorageIndexType kAvatarAccessorySlot{3};

constexpr std::chrono::milliseconds kKnockdownStunDuration{6000};
constexpr std::chrono::milliseconds kKnockbackStunDuration{2000};

} // namespace sro::game_constants

#endif // SRO_GAME_CONSTANTS_H_