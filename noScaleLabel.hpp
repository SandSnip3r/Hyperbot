#ifndef NO_SCALE_LABEL_HPP_
#define NO_SCALE_LABEL_HPP_

#include <QGraphicsSimpleTextItem>
#include <QString>
#include <QPainter>

class NoScaleLabel : public QGraphicsSimpleTextItem {
public:
  using QGraphicsSimpleTextItem::QGraphicsSimpleTextItem;
  virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
};

#endif // NO_SCALE_LABEL_HPP_
