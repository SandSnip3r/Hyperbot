#ifndef MAP_ITEM_GRAPHICS_ITEM_HPP_
#define MAP_ITEM_GRAPHICS_ITEM_HPP_

#include "ui-proto/entity.pb.h"

#include <QGraphicsItem>
#include <QPainter>

namespace map {

class ItemGraphicsItem : public QGraphicsItem {
public:
  ItemGraphicsItem(proto::entity::ItemRarity itemRarity);
  QRectF boundingRect() const override;
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
  // enum { Type = UserType + 2 };
  // int type() const override { return Type; };
private:
  static constexpr float kEntitySquareBaseRadius_{2.0};
  proto::entity::ItemRarity itemRarity_;
  void precomputeWhatToDraw();
  void updateRadius(const float newRadius);
  QPen borderPen_;
  QBrush fillBrush_;
  float shapeRadius_;
  float zoomedShapeRadius_;
};

} // namespace map

#endif // MAP_ITEM_GRAPHICS_ITEM_HPP_
