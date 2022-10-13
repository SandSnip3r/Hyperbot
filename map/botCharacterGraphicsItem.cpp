#include "botCharacterGraphicsItem.hpp"

#include <QGraphicsSceneContextMenuEvent>

namespace map {

BotCharacterGraphicsItem::BotCharacterGraphicsItem() {
  precomputeWhatToDraw();
}

void BotCharacterGraphicsItem::precomputeWhatToDraw() {
  // Border is thin
  borderPen_.setWidth(0);
  borderPen_.setColor(Qt::black);

  // Set radius first
  updateShapeZoom(1.0);

  // Tan
  circleGradient_.setColorAt(0, {255,255,128});
  circleGradient_.setColorAt(0.33, {216,158,0});
  circleGradient_.setColorAt(1, {32,16,0});
  fillBrush_ = QBrush(circleGradient_);
}

void BotCharacterGraphicsItem::updateShapeZoom(const float multiplier) {
  zoomMultiplier_ = multiplier;
  polygon_ = QPolygonF({
      QPointF{ 0*zoomMultiplier_,  12*zoomMultiplier_},
      QPointF{ 9*zoomMultiplier_,  -8*zoomMultiplier_},
      QPointF{ 0*zoomMultiplier_,  -4*zoomMultiplier_},
      QPointF{-9*zoomMultiplier_,  -8*zoomMultiplier_}});
  recalculateBoundingRect();
  if (fillBrush_.gradient() != nullptr) {
    circleGradient_.setRadius(polygon_.boundingRect().height() / 2);
    fillBrush_ = QBrush(circleGradient_);
  }
}

void BotCharacterGraphicsItem::recalculateBoundingRect() {
  prepareGeometryChange();
  // Rotate the base shape and calculate the new bounding rect
  const auto rotatedPolygon = QTransform().rotate(rotationAngleDegrees_).map(polygon_);
  const auto polygonBoundingRect = rotatedPolygon.boundingRect();
  // Add 1 to sides for pen width
  boundingRect_ = QRectF(polygonBoundingRect.left()-1, polygonBoundingRect.top()-1, polygonBoundingRect.width()+2, polygonBoundingRect.height()+2);
}

QRectF BotCharacterGraphicsItem::boundingRect() const {
  return boundingRect_;
}

void BotCharacterGraphicsItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
  if (previousM11_ != painter->worldTransform().m11()) {
    previousM11_ = painter->worldTransform().m11();
    updateShapeZoom(1 / painter->worldTransform().m11());
  }
  painter->save();
  painter->setPen(borderPen_);
  painter->setBrush(fillBrush_);

  // Don't draw the rotated polygon, draw the original one but rotate the painter
  // This will make it easier to draw additional shapes on top of the polygon
  painter->rotate(rotationAngleDegrees_);

  // Draw triangle
  painter->drawPolygon(polygon_);

  // If character is dead, draw an X through it
  // if (!alive_) {
  //   QPen pen(Qt::black);
  //   const auto penWidth = (shapeRadius_/kEntityCircleBaseRadius_) / painter->worldTransform().m11();
  //   pen.setWidthF(penWidth);
  //   painter->setPen(pen);
  //   const float x = sqrt(zoomMultiplier_*zoomMultiplier_/2);
  //   painter->drawLine(QPointF(-x,-x), QPointF(x,x));
  //   painter->drawLine(QPointF(-x,x), QPointF(x,-x));
  // }
  painter->restore();
}

void BotCharacterGraphicsItem::setDead() {
  alive_ = false;
  update();
}

void BotCharacterGraphicsItem::setAngle(float degrees) {
  rotationAngleDegrees_ = degrees;
  recalculateBoundingRect();
}

} // namespace map