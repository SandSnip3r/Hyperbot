#include "entityGraphicsItem.hpp"

#include <QGraphicsSceneContextMenuEvent>

QRectF EntityGraphicsItem::boundingRect() const {
  return QRectF(0,0,kRadius_*2,kRadius_*2);
  // return boundingRect_;
}

void EntityGraphicsItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {

  painter->save();
  QPen pen;
  pen.setWidth(0);
  painter->setPen(pen);
  painter->setBrush(QBrush(QColor(255,193,9)));
  // const auto transformedVertex = transformNavmeshCoordinateToQtCoordinate(vertex);
  painter->drawEllipse({kRadius_, kRadius_}, kRadius_, kRadius_);
  painter->restore();

  // const QColor kPathColor(0, 150, 0);
  // QPen pen(kPathColor);
  // pen.setWidthF(0.0f);

  // painter->save();
  // painter->setPen(pen);
  // painter->setRenderHint(QPainter::Antialiasing, true);
  // for (int i=0; i<shortestPath_.size(); ++i) {
  //   const pathfinder::PathSegment *segment = shortestPath_.at(i).get();
  //   const pathfinder::StraightPathSegment *straightSegment = dynamic_cast<const pathfinder::StraightPathSegment*>(segment);
  //   const pathfinder::ArcPathSegment *arcSegment = dynamic_cast<const pathfinder::ArcPathSegment*>(segment);
  //   if (straightSegment != nullptr) {
  //     if (!pathfinder::math::equal(straightSegment->startPoint.x(), straightSegment->endPoint.x()) || !pathfinder::math::equal(straightSegment->startPoint.y(), straightSegment->endPoint.y())) {
  //       // Don't want to draw a straight line that has length 0. Results in weird rendering
  //       const auto &point1 = transformNavmeshCoordinateToQtCoordinate(straightSegment->startPoint);
  //       const auto &point2 = transformNavmeshCoordinateToQtCoordinate(straightSegment->endPoint);
  //       painter->drawLine(point1, point2);
  //     } else {
  //       throw std::runtime_error("One of the line segments is 0-length. This indicates that something has gone wrong");
  //     }
  //   } else if (arcSegment != nullptr) {
  //     const auto &centerOfCircle = arcSegment->circleCenter;
  //     const auto transformedCenter = transformNavmeshCoordinateToQtCoordinate(centerOfCircle);
  //     QRectF arcRectangle(transformedCenter.x() - arcSegment->circleRadius, transformedCenter.y() - arcSegment->circleRadius, arcSegment->circleRadius*2, arcSegment->circleRadius*2);
  //     int startAngle = 360*16 * arcSegment->startAngle / pathfinder::math::k2Pi;
  //     int spanAngle = 360*16 * pathfinder::math::arcAngle(arcSegment->startAngle, arcSegment->endAngle, arcSegment->angleDirection) / pathfinder::math::k2Pi;
  //     painter->drawArc(arcRectangle, startAngle, spanAngle);
  //   }
  // }
  // painter->restore();
}

// QPointF EntityGraphicsItem::transformNavmeshCoordinateToQtCoordinate(const pathfinder::Vector &vertex) const {
//   return {vertex.x(), 1920.0f-vertex.y()};
// }
