#ifndef MAP_TRAINING_AREA_GRAPHICS_ITEM_HPP_
#define MAP_TRAINING_AREA_GRAPHICS_ITEM_HPP_

#include "proto/broadcast.pb.h"

#include <QGraphicsItem>
#include <QPainter>

namespace map {

class TrainingAreaGraphicsItem : public QGraphicsItem {
public:
  TrainingAreaGraphicsItem(float radius);
  QRectF boundingRect() const override;
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
  // enum { Type = UserType + 2 };
  // int type() const override { return Type; };
private:
  const float radius_;
};

} // namespace map

#endif // MAP_TRAINING_AREA_GRAPHICS_ITEM_HPP_
