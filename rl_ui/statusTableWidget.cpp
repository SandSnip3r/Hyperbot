#include "statusTableWidget.hpp"

#include <QApplication>
#include <QDrag>
#include <QMimeData>
#include <QMouseEvent>

StatusTableWidget::StatusTableWidget(QWidget *parent)
    : QTableWidget(parent) {
  setDragEnabled(true);
}

void StatusTableWidget::mousePressEvent(QMouseEvent *event) {
  dragStartPos_ = event->pos();
  QTableWidget::mousePressEvent(event);
}

void StatusTableWidget::mouseMoveEvent(QMouseEvent *event) {
  if (!(event->buttons() & Qt::LeftButton)) {
    QTableWidget::mouseMoveEvent(event);
    return;
  }
  if ((event->pos() - dragStartPos_).manhattanLength() <
      QApplication::startDragDistance()) {
    QTableWidget::mouseMoveEvent(event);
    return;
  }
  int row = rowAt(dragStartPos_.y());
  if (row < 0) {
    return;
  }
  QTableWidgetItem *item = this->item(row, 0);
  if (!item) {
    return;
  }
  QString name = item->text();
  QMimeData *mimeData = new QMimeData;
  mimeData->setText(name);

  QDrag *drag = new QDrag(this);
  drag->setMimeData(mimeData);
  drag->exec(Qt::CopyAction);
}

