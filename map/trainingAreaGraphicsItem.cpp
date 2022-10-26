#include "trainingAreaGraphicsItem.hpp"

#include <QGraphicsSceneContextMenuEvent>

namespace map {

TrainingAreaGraphicsItem::TrainingAreaGraphicsItem(float radius) : radius_(radius) {
}

QRectF TrainingAreaGraphicsItem::boundingRect() const {
  return {-radius_, -radius_, radius_*2, radius_*2};
}

void TrainingAreaGraphicsItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
  painter->save();
  painter->setPen(Qt::NoPen);
  QBrush brush(QColor(128, 128, 255, 75));
  painter->setBrush(brush);

  // Draw circle
  painter->drawEllipse(QPointF(0,0), radius_, radius_);

  painter->restore();
}

} // namespace map