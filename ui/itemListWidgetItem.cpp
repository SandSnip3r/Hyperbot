#include "itemListWidgetItem.hpp"

ItemListWidgetItem::ItemListWidgetItem(const uint8_t slotIndex, const uint16_t quantity, const std::string &itemName, QListWidget *parent) : slotIndex_(slotIndex), quantity_(quantity), itemName_(itemName), QListWidgetItem(parent, Type) {}

ItemListWidgetItem& ItemListWidgetItem::operator=(const ItemListWidgetItem &other) {
  slotIndex_ = other.slotIndex_;
  quantity_ = other.quantity_;
  itemName_ = other.itemName_;
  return *this;
}

QVariant ItemListWidgetItem::data(int role) const {
  if (role == Qt::DisplayRole) {
    return toString();
  } else {
    return {};
  }
}

QString ItemListWidgetItem::toString() const {
  QString result;
  result.append(QString("%1 - ").arg(slotIndex_,3));
  result.append(QString::fromStdString(itemName_));
  if (quantity_ != 1) {
    result.append(QString(" x %1").arg(quantity_));
  }
  return result;
}
