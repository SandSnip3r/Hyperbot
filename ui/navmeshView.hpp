#ifndef NAVMESHVIEW_H
#define NAVMESHVIEW_H

#include <silkroad_lib/math/vector3.hpp>
#include <silkroad_lib/navmesh/triangulation/navmeshTriangulation.hpp>

#include <QAction>
#include <QFrame>
#include <QGraphicsView>
#include <QWheelEvent>

#include <iostream>

class NavmeshView;

class NavmeshGraphicsView : public QGraphicsView {
  Q_OBJECT
public:
  NavmeshGraphicsView(QWidget *parent=nullptr);
  void setWorldNavmesh(const sro::navmesh::triangulation::NavmeshTriangulation &navmeshTriangulation);

protected:
  void wheelEvent(QWheelEvent *event) override;
  virtual void contextMenuEvent(QContextMenuEvent *event);
  virtual void mouseMoveEvent(QMouseEvent *event) override;

public:
  void updateLabels();

private:
  NavmeshView *navmeshView_;
  const sro::navmesh::triangulation::NavmeshTriangulation *navmeshTriangulation_{nullptr};
  const qreal kMinimumZoomForLabels_{1.0};
  bool showTriangleLabels_{false};
  bool showEdgeLabels_{false};
  bool showVertexLabels_{false};

signals:
  void setPathStart(const sro::math::Vector3 &pos, const uint16_t regionId);
  void setPathGoal(const sro::math::Vector3 &pos, const uint16_t regionId);
  void resetPath();
  void mouseMoved(float absoluteX, float absoluteY, int regionX, int regionY, uint16_t regionId, float xInRegion, float yInRegion);

public slots:
  void showTriangleLabelsSettingToggled(bool isSet);
  void showEdgeLabelsSettingToggled(bool isSet);
  void showVertexLabelsSettingToggled(bool isSet);
};

class NavmeshView : public QFrame {
  Q_OBJECT
public:
  using QFrame::QFrame;
  void setNavmeshGraphicsView(NavmeshGraphicsView *navmeshGraphicsView);

  QGraphicsView* getView() const;
  void zoomIn(double diff);
  void zoomOut(double diff);
  double getZoomLevel() const;
private:
  NavmeshGraphicsView *navmeshGraphicsView_{nullptr};
  double zoomLevel_{0};

  void setupMatrix();
};

#endif // NAVMESHVIEW_H
