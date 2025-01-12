#ifndef PACKET_LIST_WIDGET_ITEM_HPP_
#define PACKET_LIST_WIDGET_ITEM_HPP_

#include "ui-proto/request.pb.h"

#include <QListWidgetItem>

class PacketListWidgetItem : public QListWidgetItem {
public:
  PacketListWidgetItem(proto::request::PacketToInject::Direction dir, const uint16_t op, std::string d, QListWidget *parent = nullptr);
  virtual QVariant data(int role) const override;
  proto::request::PacketToInject::Direction direction() const;
  uint16_t opcode() const;
  std::string data() const;
  enum { Type = UserType + 1 };
private:
  proto::request::PacketToInject::Direction direction_;
  const uint16_t opcode_;
  std::string data_;

  QString toString() const;
};

#endif // PACKET_LIST_WIDGET_ITEM_HPP_
