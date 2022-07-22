#ifndef HELPERS_HPP_
#define HELPERS_HPP_

#include "packet/enums/packetEnums.hpp"
#include "packet/structures/packetInnerStructures.hpp"
#include "pk2/gameData.hpp"
#include "state/entity.hpp"
#include "storage/item.hpp"
#include "storage/storage.hpp"

#include "../../common/pk2/ref/item.hpp"
#include "../../common/pk2/ref/scrapOfPackageItem.hpp"

#include <map>
#include <memory>

namespace helpers {

float secondsToTravel(const packet::structures::Position &srcPosition, const packet::structures::Position &destPosition, const float currentSpeed);
void initializeInventory(storage::Storage &inventory, uint8_t inventorySize, const std::map<uint8_t, std::shared_ptr<storage::Item>> &inventoryItemMap);
void printItem(uint8_t slot, const storage::Item *item, const pk2::GameData &gameData);
int toBitNum(packet::enums::AbnormalStateFlag stateFlag);
packet::enums::AbnormalStateFlag fromBitNum(int n);
std::shared_ptr<storage::Item> createItemFromScrap(const pk2::ref::ScrapOfPackageItem &itemScrap, const pk2::ref::Item &itemRef);
void trackObject(state::Entity &entityState, std::shared_ptr<packet::parsing::Object> obj);
void stopTrackingObject(state::Entity &entityState, uint32_t gId);

namespace type_id {

// TODO: Create a more elegant TypeId system
uint16_t makeTypeId(const uint16_t typeId1, const uint16_t typeId2, const uint16_t typeId3, const uint16_t typeId4);
bool isUniversalPill(const pk2::ref::Item &itemInfo);
bool isPurificationPill(const pk2::ref::Item &itemInfo);
bool isHpPotion(const pk2::ref::Item &itemInfo);
bool isMpPotion(const pk2::ref::Item &itemInfo);
bool isVigorPotion(const pk2::ref::Item &itemInfo);

} // namespace type_id

} // namespace helpers

#endif // HELPERS_HPP_