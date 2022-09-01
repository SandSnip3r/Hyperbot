#ifndef ITEM_LIST_WIDGET_HPP_
#define ITEM_LIST_WIDGET_HPP_

#include "itemListWidgetItem.hpp"

#include <QListWidget>
#include <QObject>

class ItemListWidget : public QListWidget {
public:
  using QListWidget::QListWidget;
  void addItem(ItemListWidgetItem *itemToAdd);
  void removeItem(uint8_t slotIndex);
};

#endif // ITEM_LIST_WIDGET_HPP_
