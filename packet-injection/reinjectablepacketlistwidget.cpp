#include "reinjectablepacketlistwidget.h"

#include <QContextMenuEvent>
#include <QMenu>

void ReinjectablePacketListWidget::contextMenuEvent(QContextMenuEvent *event) {
  if (event->reason() != QContextMenuEvent::Mouse) {
    // Only care about mouse right clicks
    return;
  }

  QMenu *menu = new QMenu(this);

  if (!selectedItems().empty()) {
    // Create an action to reset the path
    QAction *action = new QAction(QString("Reinject selected packets"), menu);
    connect(action, &QAction::triggered, this, [=]() {
      emit reinjectSelectedPackets();
    });
    menu->addAction(action);

    menu->addSeparator();
  }

  QAction *clearAction = new QAction(QString("Clear"), menu);
  connect(clearAction, &QAction::triggered, this, [=]() {
    emit clearPackets();
  });
  menu->addAction(clearAction);

  // Display menu asynchronously
  menu->popup(event->globalPos());
}
