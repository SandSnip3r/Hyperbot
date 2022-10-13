#ifndef MAP_CHARACTER_GRAPHICS_ITEM_HPP_
#define MAP_CHARACTER_GRAPHICS_ITEM_HPP_

#include "proto/broadcast.pb.h"

#include <QGraphicsItem>
#include <QPainter>

namespace map {

class CharacterGraphicsItem : public QGraphicsItem {
public:
  CharacterGraphicsItem(broadcast::EntityType type);
  QRectF boundingRect() const override;
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
  // enum { Type = UserType + 2 };
  // int type() const override { return Type; };
  void setDead();
private:
  static constexpr float kEntityCircleBaseRadius_{5.0};
  broadcast::EntityType entityType_;
  void precomputeWhatToDraw();
  void updateRadius(const float newRadius);
  QPen borderPen_;
  QBrush fillBrush_;
  QRadialGradient circleGradient_;
  float shapeRadius_;
  float zoomedShapeRadius_;
  bool alive_{true};
};

} // namespace map

#endif // MAP_CHARACTER_GRAPHICS_ITEM_HPP_
