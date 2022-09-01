#include "packetListWidgetItem.hpp"

PacketListWidgetItem::PacketListWidgetItem(request::PacketToInject::Direction dir, const uint16_t op, std::string d, QListWidget *parent) : direction_(dir), opcode_(op), data_(d), QListWidgetItem(parent, Type) {
  //
}

QVariant PacketListWidgetItem::data(int role) const {
  if (role == Qt::DisplayRole) {
    return toString();
  } else {
    return {};
  }
}

request::PacketToInject::Direction PacketListWidgetItem::direction() const {
  return direction_;
}

uint16_t PacketListWidgetItem::opcode() const {
  return opcode_;
}

std::string PacketListWidgetItem::data() const {
  return data_;
}

QString PacketListWidgetItem::toString() const {
  QString result;
  if (direction_ == request::PacketToInject::kClientToServer) {
    result = "[C->S] ";
  } else {
    result = "[S->C] ";
  }
  result.append(QString("%1 - ").arg(opcode_,4,16,QChar('0')));
  for (const auto i : data_) {
    result.append(QString("%1 ").arg(static_cast<uint8_t>(i),2,16,QChar('0')));
  }
  return result;
}