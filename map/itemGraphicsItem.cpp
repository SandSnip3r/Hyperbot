#include "itemGraphicsItem.hpp"

#include <QGraphicsSceneContextMenuEvent>

namespace map {

ItemGraphicsItem::ItemGraphicsItem(broadcast::EntityType type) : entityType_(type) {
  precomputeWhatToDraw();
}

void ItemGraphicsItem::precomputeWhatToDraw() {
  // All things are drawn with a thin black border
  borderPen_.setWidth(0);
  borderPen_.setColor({0,0,0});
  fillBrush_.setStyle(Qt::BrushStyle::SolidPattern);
  if (entityType_ == broadcast::EntityType::kItemCommon) {
    fillBrush_.setColor({255,255,255});
  } else if (entityType_ == broadcast::EntityType::kItemRare) {
    fillBrush_.setColor({114,191,255});
  } else if (entityType_ == broadcast::EntityType::kItemSox) {
    fillBrush_.setColor({255,217,83});
  }
  shapeRadius_ = kEntitySquareBaseRadius_;
  updateRadius(shapeRadius_);
}

void ItemGraphicsItem::updateRadius(const float newRadius) {
  zoomedShapeRadius_ = newRadius;
}

QRectF ItemGraphicsItem::boundingRect() const {
  return QRectF(-zoomedShapeRadius_-1, -zoomedShapeRadius_-1, zoomedShapeRadius_*2+1, zoomedShapeRadius_*2+1);
}

void ItemGraphicsItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
  static auto previousM11 = painter->worldTransform().m11();
  if (previousM11 != painter->worldTransform().m11()) {
    previousM11 = painter->worldTransform().m11();
    prepareGeometryChange();
  }
  updateRadius(shapeRadius_ * 1/painter->worldTransform().m11());
  painter->save();
  painter->setPen(borderPen_);
  painter->setBrush(fillBrush_);
  // Draw square
  painter->drawRect(QRectF(-zoomedShapeRadius_, -zoomedShapeRadius_, 2*zoomedShapeRadius_, 2*zoomedShapeRadius_));
  painter->restore();
}

} // namespace map