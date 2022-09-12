#ifndef ENTITY_GRAPHICS_ITEM_HPP_
#define ENTITY_GRAPHICS_ITEM_HPP_

// #include "Pathfinder/pathfinder.h"

#include <QGraphicsItem>
#include <QPainter>
// #include <QStyleOptionGraphicsItem>

class EntityGraphicsItem : public QGraphicsItem {
public:
  EntityGraphicsItem() = default;
  QRectF boundingRect() const override;
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
  // enum { Type = UserType + 2 };
  // int type() const override { return Type; };
private:
  static constexpr const float kRadius_{10.0};
  // const std::vector<std::unique_ptr<pathfinder::PathSegment>> &shortestPath_;
  // QRectF boundingRect_;
  // void setupBoundingRect();
  // QPointF transformNavmeshCoordinateToQtCoordinate(const pathfinder::Vector &vertex) const;
};

#endif // ENTITY_GRAPHICS_ITEM_HPP_
