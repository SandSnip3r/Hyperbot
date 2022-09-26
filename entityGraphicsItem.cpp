#include "entityGraphicsItem.hpp"

#include <QGraphicsSceneContextMenuEvent>

EntityGraphicsItem::EntityGraphicsItem(sro::types::EntityType type) : entityType_(type) {
}

QRectF EntityGraphicsItem::boundingRect() const {
  return QRectF(-pointRadius_, -pointRadius_, pointRadius_*2, pointRadius_*2);
}

void EntityGraphicsItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
  painter->save();
  QPen pen;
  pen.setWidth(0);
  painter->setPen(pen);
  switch (entityType_) {
    case sro::types::EntityType::kSelf:
      painter->setBrush(QBrush(QColor(255,193,9)));
      break;
    case sro::types::EntityType::kPlayerCharacter:
      painter->setBrush(QBrush(QColor(204,239,57)));
      break;
    case sro::types::EntityType::kNonplayerCharacter:
      painter->setBrush(QBrush(QColor(53,107,230)));
      break;
    case sro::types::EntityType::kMonster:
      painter->setBrush(QBrush(QColor(255,1,1)));
      break;
    case sro::types::EntityType::kItem:
      painter->setBrush(QBrush(QColor(255,255,255)));
      break;
    case sro::types::EntityType::kCharacter:
    case sro::types::EntityType::kPortal:
    default:
      painter->setBrush(QBrush(QColor(175,0,175)));
      break;
  }
  // Update radius of point based on zoom level
  pointRadius_ = 3.0 * 1/painter->worldTransform().m11();
  // Draw point
  painter->drawEllipse({0, 0}, pointRadius_, pointRadius_);
  painter->restore();
}
