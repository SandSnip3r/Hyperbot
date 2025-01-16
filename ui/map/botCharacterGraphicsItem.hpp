#ifndef MAP_BOT_CHARACTER_GRAPHICS_ITEM_HPP_
#define MAP_BOT_CHARACTER_GRAPHICS_ITEM_HPP_

#include "ui_proto/broadcast.pb.h"

#include <QGraphicsItem>
#include <QPainter>

namespace map {

class BotCharacterGraphicsItem : public QGraphicsItem {
public:
  BotCharacterGraphicsItem();
  QRectF boundingRect() const override;
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
  // enum { Type = UserType + 2 };
  // int type() const override { return Type; };
  void setDead();
  void setAngle(float degrees);
private:
  static constexpr float kEntityCircleBaseRadius_{1.0};
  void precomputeWhatToDraw();
  void updateShapeZoom(const float multiplier);
  void recalculateBoundingRect();
  QPen borderPen_;
  QBrush fillBrush_;
  QPolygonF polygon_;
  QRectF boundingRect_;
  QRadialGradient circleGradient_;
  float zoomMultiplier_;
  bool alive_{true};
  qreal rotationAngleDegrees_{0.0};
  qreal previousM11_{0.0};
};

} // namespace map

#endif // MAP_BOT_CHARACTER_GRAPHICS_ITEM_HPP_
