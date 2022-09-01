#include "itemListWidget.hpp"

void ItemListWidget::addItem(ItemListWidgetItem *itemToAdd) {
  // Figure out where to put this
  for (int row=0; row<count(); ++row) {
    auto *itemAtThisRow = dynamic_cast<ItemListWidgetItem*>(this->item(row));
    if (itemAtThisRow->slotIndex_ == itemToAdd->slotIndex_) {
      // This is an overwrite!
      *itemAtThisRow = *itemToAdd;
      delete itemToAdd;
      QListWidget::viewport()->update(); // This seems necessary
      // I also tried removing and adding the new item, but that resulted in a weird UI flicker
      return;
    } else if (itemAtThisRow->slotIndex_ > itemToAdd->slotIndex_) {
      // This is the place to put it
      insertItem(row, itemToAdd);
      return;
    }
  }

  // List is empty, or we didnt find a place to put this item (which means this is the last item so far)
  QListWidget::addItem(itemToAdd);
}

void ItemListWidget::removeItem(uint8_t slotIndex) {
  for (int row=0; row<count(); ++row) {
    auto *item = dynamic_cast<ItemListWidgetItem*>(this->item(row));
    if (item->slotIndex_ == slotIndex) {
      // This is the item to remove
      delete item;
      return;
    }
  }
}
