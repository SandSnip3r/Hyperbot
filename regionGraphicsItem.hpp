#ifndef REGIONGRAPHICSITEM_H
#define REGIONGRAPHICSITEM_

#include <silkroad_lib/navmesh/navmesh.h>
#include <silkroad_lib/navmesh/triangulation/singleRegionNavmeshTriangulation.h>

#include <QGraphicsItem>
#include <QGraphicsSimpleTextItem>
#include <QPainter>
#include <QPixmap>
#include <QStyleOptionGraphicsItem>

#include <atomic>
#include <memory>

class NoScaleLabel;

enum class LabelStatus {
  kNotAdded,
  kAdded,
  kNeedToRemove,
  kNeedToAdd
};

class RegionGraphicsItem : public QGraphicsItem {
public:
  RegionGraphicsItem(const sro::navmesh::Navmesh &navmesh, const sro::navmesh::Region &region, const sro::navmesh::triangulation::SingleRegionNavmeshTriangulation &navmeshTriangulation);
  ~RegionGraphicsItem();
  QRectF boundingRect() const override;
  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
  const sro::navmesh::Navmesh& getNavmesh() const;
  const sro::navmesh::Region& getRegion() const;
  const sro::navmesh::triangulation::SingleRegionNavmeshTriangulation& getNavmeshTriangulation() const;
  enum { Type = UserType + 1 };
  int type() const override { return Type; };
  void addTriangleLabels();
  void addEdgeLabels();
  void addVertexLabels();
  void removeTriangleLabels();
  void removeEdgeLabels();
  void removeVertexLabels();
  void createLabels();
  void setPixmap(const QPixmap &pixmap);
private:
  const sro::navmesh::Navmesh &navmesh_;
  const sro::navmesh::Region &region_;
  const sro::navmesh::triangulation::SingleRegionNavmeshTriangulation &navmeshTriangulation_;
  static std::map<sro::navmesh::triangulation::ObjectData, QColor> objectColorMap_;
  mutable std::map<uint32_t, QColor> linkColorMap_;
  std::atomic<bool> labelsCreated_{false};
  LabelStatus triangleLabelStatus_{LabelStatus::kNotAdded}, edgeLabelStatus_{LabelStatus::kNotAdded}, vertexLabelStatus_{LabelStatus::kNotAdded};
  std::vector<NoScaleLabel*> triangleLabels_, edgeLabels_, vertexLabels_;
  QPixmap pixmap_;

  void drawBlockedTiles(QPainter &painter, const QStyleOptionGraphicsItem &option) const;
  void drawObjectColors(QPainter &painter, const QStyleOptionGraphicsItem &option) const;
  void drawLinkAreas(QPainter &painter, const QStyleOptionGraphicsItem &option) const;
  void drawVertices(QPainter &painter, const QStyleOptionGraphicsItem &option) const;
  void drawEdges(QPainter &painter, const QStyleOptionGraphicsItem &option) const;
  void drawEdgeLinks(QPainter &painter, const QStyleOptionGraphicsItem &option) const;
  void addOrRemoveLabelsIfNeccessary();

  QColor getColorForEdgeMarker(const int marker) const;

  QPointF transformNavmeshCoordinateToQtCoordinate(const pathfinder::Vector &vertex) const;
};

#endif // REGIONGRAPHICSITEM_H
