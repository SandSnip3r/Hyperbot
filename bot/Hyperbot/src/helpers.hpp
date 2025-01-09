#ifndef HELPERS_HPP_
#define HELPERS_HPP_

#include "packet/enums/packetEnums.hpp"
#include "packet/structures/packetInnerStructures.hpp"
#include "pk2/gameData.hpp"
#include "storage/item.hpp"
#include "storage/storage.hpp"

#include "../../common/pk2/ref/item.hpp"
#include "../../common/pk2/ref/scrapOfPackageItem.hpp"

#include <silkroad_lib/position.h>

#include <filesystem>
#include <map>
#include <memory>
#include <tuple>

namespace helpers {

std::filesystem::path getAppDataDirectory();
float secondsToTravel(const sro::Position &srcPosition, const sro::Position &destPosition, const float currentSpeed);
void initializeInventory(storage::Storage &inventory, uint8_t inventorySize, const std::map<uint8_t, std::shared_ptr<storage::Item>> &inventoryItemMap);
void printItem(uint8_t slot, const storage::Item *item, const pk2::GameData &gameData);
int toBitNum(packet::enums::AbnormalStateFlag stateFlag);
packet::enums::AbnormalStateFlag fromBitNum(int n);
std::shared_ptr<storage::Item> createItemFromScrap(const pk2::ref::ScrapOfPackageItem &itemScrap, const pk2::ref::Item &itemRef);

namespace type_id {

// TODO: Move uses of this type Id stuff to use the new type id categories system
std::tuple<uint8_t,uint8_t,uint8_t,uint8_t> splitTypeId(const uint16_t typeId);

} // namespace type_id

} // namespace helpers

#endif // HELPERS_HPP_