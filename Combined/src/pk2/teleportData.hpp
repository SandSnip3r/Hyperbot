#ifndef PK2_MEDIA_TELEPORT_DATA_HPP
#define PK2_MEDIA_TELEPORT_DATA_HPP

#include "../../../common/pk2/ref/teleport.hpp"

#include <unordered_map>

namespace pk2 {

class TeleportData {
public:
	using TeleportMap = std::unordered_map<ref::TeleportId,ref::Teleport>;
	void addTeleport(ref::Teleport &&teleport);
	bool haveTeleportWithId(ref::TeleportId id) const;
	const ref::Teleport& getTeleportById(ref::TeleportId id) const;
	const TeleportMap::size_type size() const;
private:
	TeleportMap teleports_;
};

} // namespace pk2

#endif // PK2_MEDIA_TELEPORT_DATA_HPP