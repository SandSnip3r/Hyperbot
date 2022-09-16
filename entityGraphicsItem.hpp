#ifndef ENTITY_GRAPHICS_ITEM_HPP_
#define ENTITY_GRAPHICS_ITEM_HPP_

#include <silkroad_lib/entity_types.h>

#include <QGraphicsItem>
#include <QPainter>

class EntityGraphicsItem : public QGraphicsItem {
public:
  EntityGraphicsItem(sro::entity_types::EntityType type);
  QRectF boundingRect() const override;
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
  // enum { Type = UserType + 2 };
  // int type() const override { return Type; };
private:
  double pointRadius_{3.0};
  sro::entity_types::EntityType entityType_;
};

#endif // ENTITY_GRAPHICS_ITEM_HPP_
