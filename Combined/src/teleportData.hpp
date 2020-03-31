#ifndef PK2_MEDIA_TELEPORT_DATA_HPP
#define PK2_MEDIA_TELEPORT_DATA_HPP

#include "../../common/teleport.hpp"

#include <unordered_map>

namespace pk2::media {

class TeleportData {
public:
	using TeleportMap = std::unordered_map<TeleportId,Teleport>;
	void addTeleport(Teleport &&teleport);
	bool haveTeleportWithId(TeleportId id) const;
	const Teleport& getTeleportById(TeleportId id) const;
	const TeleportMap::size_type size() const;
private:
	TeleportMap teleports_;
};

} // namespace pk2::media

#endif // PK2_MEDIA_TELEPORT_DATA_HPP