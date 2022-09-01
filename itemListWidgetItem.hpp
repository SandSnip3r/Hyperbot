#ifndef ITEM_LIST_WIDGET_ITEM_HPP_
#define ITEM_LIST_WIDGET_ITEM_HPP_

#include <QListWidgetItem>

#include <cstdint>
#include <string>

class ItemListWidget;

class ItemListWidgetItem : public QListWidgetItem {
public:
  ItemListWidgetItem(const uint8_t slotIndex, const uint16_t quantity, const std::string &itemName, QListWidget *parent = nullptr);
  ItemListWidgetItem& operator=(const ItemListWidgetItem &other);
  virtual QVariant data(int role) const override;
  enum { Type = UserType + 2 };
private:
  uint8_t slotIndex_;
  uint16_t quantity_;
  std::string itemName_;
  QString toString() const;
  friend class ItemListWidget;
};

#endif // ITEM_LIST_WIDGET_ITEM_HPP_
