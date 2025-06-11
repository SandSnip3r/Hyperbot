#ifndef PK2_MEDIA_TELEPORT_DATA_HPP
#define PK2_MEDIA_TELEPORT_DATA_HPP

#include <silkroad_lib/pk2/ref/teleport.hpp>

#include <unordered_map>

namespace sro::pk2 {

class TeleportData {
public:
	using TeleportMap = std::unordered_map<sro::pk2::ref::TeleportId,sro::pk2::ref::Teleport>;
	void addTeleport(sro::pk2::ref::Teleport &&teleport);
	bool haveTeleportWithId(sro::pk2::ref::TeleportId id) const;
	const sro::pk2::ref::Teleport& getTeleportById(sro::pk2::ref::TeleportId id) const;
	const TeleportMap::size_type size() const;
private:
	TeleportMap teleports_;
};

} // namespace sro::pk2

#endif // PK2_MEDIA_TELEPORT_DATA_HPP