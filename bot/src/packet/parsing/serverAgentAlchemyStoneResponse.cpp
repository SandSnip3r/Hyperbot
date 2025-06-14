#include "serverAgentAlchemyStoneResponse.hpp"

#include "packet/enums/packetEnums.hpp"
#include "packet/parsing/commonParsing.hpp"

#include <silkroad_lib/scalar_types.hpp>

namespace packet::parsing {

ServerAgentAlchemyStoneResponse::ServerAgentAlchemyStoneResponse(const PacketContainer &packet, const sro::pk2::ItemData &itemData) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  stream.Read(result_);
  if (result_ == 1) {
    stream.Read(alchemyAction_);
    if (alchemyAction_ == packet::enums::AlchemyAction::kCancel) {
      // Done.
      return;
    }

    stream.Read(success_);
    stream.Read(slot_);
    // bool itemWasDestroyed;
    // if (!success_) {
    //   stream.Read(itemWasDestroyed_);
    // }

    // if (!itemWasDestroyed_) {
      item_ = parseGenericItem(stream, itemData);
    // }
  } else if (result_ == 2) {
    stream.Read(errorCode_);
  }
}

uint8_t ServerAgentAlchemyStoneResponse::result() const {
  return result_;
}
packet::enums::AlchemyAction ServerAgentAlchemyStoneResponse::alchemyAction() const {
  return alchemyAction_;
}
bool ServerAgentAlchemyStoneResponse::success() const {
  return success_;
}
sro::scalar_types::StorageIndexType ServerAgentAlchemyStoneResponse::slot() const {
  return slot_;
}
std::shared_ptr<storage::Item> ServerAgentAlchemyStoneResponse::item() const {
  return item_;
}

uint16_t ServerAgentAlchemyStoneResponse::errorCode() const {
  return errorCode_;
}

} // namespace packet::parsing