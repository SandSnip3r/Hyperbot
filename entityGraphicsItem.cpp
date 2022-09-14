#include "entityGraphicsItem.hpp"

#include <QGraphicsSceneContextMenuEvent>

EntityGraphicsItem::EntityGraphicsItem(sro::entity_types::EntityType type) : entityType_(type) {
}

QRectF EntityGraphicsItem::boundingRect() const {
  return QRectF(0,0,kRadius_*2,kRadius_*2);
}

void EntityGraphicsItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
  painter->save();
  QPen pen;
  pen.setWidth(0);
  painter->setPen(pen);
  switch (entityType_) {
    case sro::entity_types::EntityType::kSelf:
      painter->setBrush(QBrush(QColor(255,193,9)));
      break;
    case sro::entity_types::EntityType::kPlayerCharacter:
      painter->setBrush(QBrush(QColor(71,255,71)));
      break;
    case sro::entity_types::EntityType::kNonplayerCharacter:
      painter->setBrush(QBrush(QColor(53,107,230)));
      break;
    case sro::entity_types::EntityType::kMonster:
      painter->setBrush(QBrush(QColor(255,1,1)));
      break;
    case sro::entity_types::EntityType::kItem:
      painter->setBrush(QBrush(QColor(255,255,255)));
      break;
    case sro::entity_types::EntityType::kCharacter:
    case sro::entity_types::EntityType::kPortal:
    default:
      painter->setBrush(QBrush(QColor(175,0,175)));
      break;
  }
  painter->drawEllipse({kRadius_, kRadius_}, kRadius_, kRadius_);
  painter->restore();
}
