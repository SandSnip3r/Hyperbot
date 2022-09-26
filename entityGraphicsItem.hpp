#ifndef ENTITY_GRAPHICS_ITEM_HPP_
#define ENTITY_GRAPHICS_ITEM_HPP_

#include "sro_types.hpp"

#include <QGraphicsItem>
#include <QPainter>

class EntityGraphicsItem : public QGraphicsItem {
public:
  EntityGraphicsItem(sro::types::EntityType type);
  QRectF boundingRect() const override;
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
  // enum { Type = UserType + 2 };
  // int type() const override { return Type; };
private:
  double pointRadius_{3.0};
  sro::types::EntityType entityType_;
};

#endif // ENTITY_GRAPHICS_ITEM_HPP_
