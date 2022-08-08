#ifndef REINJECTABLE_PACKET_LIST_WIDGET_HPP_
#define REINJECTABLE_PACKET_LIST_WIDGET_HPP_

#include <QListWidget>

class ReinjectablePacketListWidget : public QListWidget {
  Q_OBJECT
public:
  using QListWidget::QListWidget;
protected:
  virtual void contextMenuEvent(QContextMenuEvent *event) override;
signals:
  void reinjectSelectedPackets();
  void clearPackets();
};

#endif // REINJECTABLE_PACKET_LIST_WIDGET_HPP_
