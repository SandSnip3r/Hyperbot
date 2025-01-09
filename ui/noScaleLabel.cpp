#include "noScaleLabel.hpp"

#include <QStyleOptionGraphicsItem>

#include <iostream>

void NoScaleLabel::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
  const QTransform t = painter->transform();
  painter->setTransform(QTransform(1,       t.m12(), t.m13(),
                                   t.m21(), 1,       t.m23(),
                                   t.m31(), t.m32(), t.m33()));
  QGraphicsSimpleTextItem::paint(painter, option, widget);
  painter->setTransform(t);
}
