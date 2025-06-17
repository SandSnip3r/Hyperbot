#include "draggableTableWidget.hpp"

#include <QApplication>
#include <QDrag>
#include <QMimeData>
#include <QMouseEvent>

void DraggableTableWidget::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    dragStartPos_ = event->pos();
  }
  QTableWidget::mousePressEvent(event);
}

void DraggableTableWidget::mouseMoveEvent(QMouseEvent *event) {
  if (!(event->buttons() & Qt::LeftButton)) {
    QTableWidget::mouseMoveEvent(event);
    return;
  }
  if ((event->pos() - dragStartPos_).manhattanLength() <
      QApplication::startDragDistance()) {
    QTableWidget::mouseMoveEvent(event);
    return;
  }
  QTableWidgetItem *item = itemAt(dragStartPos_);
  if (!item) {
    return;
  }
  QMimeData *mimeData = new QMimeData;
  mimeData->setText(item->text());
  QDrag *drag = new QDrag(this);
  drag->setMimeData(mimeData);
  drag->exec(Qt::CopyAction);
}
