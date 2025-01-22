#include "navmeshView.hpp"
#include "regionGraphicsItem.hpp"

#include <silkroad_lib/position_math.hpp>
#include <silkroad_lib/math/matrix4x4.hpp>

#include <QGridLayout>
#include <QMenu>
#include <QOpenGLWidget>
#include <QtMath>

#include <vector>

NavmeshGraphicsView::NavmeshGraphicsView(QWidget *parent) : QGraphicsView(parent), navmeshView_(dynamic_cast<NavmeshView*>(parent)) {
  if (navmeshView_ == nullptr) {
    throw std::runtime_error("NavmeshGraphicsView constructed with a parent that is not a NavmeshView");
  }
  navmeshView_->setNavmeshGraphicsView(this);
  setMouseTracking(true);
  setDragMode(QGraphicsView::ScrollHandDrag);
  setInteractive(false);
  // setViewport(new QOpenGLWidget);
  // setBackgroundBrush(QBrush(Qt::white));
}

void NavmeshGraphicsView::setWorldNavmesh(const sro::navmesh::triangulation::NavmeshTriangulation &navmeshTriangulation) {
  navmeshTriangulation_ = &navmeshTriangulation;
}

void NavmeshGraphicsView::wheelEvent(QWheelEvent *event) {
  if (event->modifiers() & Qt::ControlModifier) {
    if (event->angleDelta().y() > 0) {
      navmeshView_->zoomIn(0.15);
    } else {
      navmeshView_->zoomOut(0.15);
    }
    event->accept();
  } else {
    QGraphicsView::wheelEvent(event);
  }
}

void NavmeshGraphicsView::contextMenuEvent(QContextMenuEvent *event) {
  if (event->reason() != QContextMenuEvent::Mouse) {
    // Only care about mouse right clicks
    return;
  }
  // Map mouse click position to Silkroad world position
  const auto transformedPos = mapToScene(event->pos());
  const int regionX = transformedPos.x()/1920.0;
  const int regionY = 128 - transformedPos.y()/1920.0 + 1;
  const auto regionId = sro::position_math::worldRegionIdFromSectors(regionX,regionY);
  const float x = fmod(transformedPos.x(),1920.0);
  const float y = 1920.0 - fmod(transformedPos.y(),1920.0);
  const int sroX = (regionX-135)*192+x/10;
  const int sroY = (regionY-92)*192+y/10;
  // Get the region GraphicsItem where the user clicked
  const auto itemsAtPos = items(event->pos());
  if (itemsAtPos.empty()) {
    // There are no items here
    return;
  }

  // Get the last item in the list (this is guaranteed to be the bottom item, aka first added, aka navmesh)
  const auto *regionItem = qgraphicsitem_cast<const RegionGraphicsItem*>(itemsAtPos.back());
  if (regionItem == nullptr) {
    // There is no region here, I dont think there's anything to do at this point
    return;
  }

  // Get all of the possible surfaces for this x,y and populate a context menu
  const auto triangleIndex = regionItem->getNavmeshTriangulation().findTriangleForPoint({static_cast<float>(x), static_cast<float>(y)});
  if (!triangleIndex) {
    throw std::runtime_error("Point in region has no valid triangle");
  }

  constexpr const auto kTerrainDestinationName{"terrain"};
  struct Destination {
    enum class Tag {
      kStart,
      kGoal
    };
    Destination(const std::string &n, const sro::math::Vector3 &p, const Tag &t) : name(n), pos(p), tag(t) {}
    std::string name;
    sro::math::Vector3 pos;
    Tag tag;
  };

  std::vector<Destination> destinations;

  // First, look at the terrain
  const bool blocked = regionItem->getNavmeshTriangulation().terrainIsBlockedUnderTriangle(*triangleIndex);
  if (!blocked) {
    const float heightOnTerrain = regionItem->getRegion().getHeightAtPoint({static_cast<float>(x), 0.0f, static_cast<float>(y)});
    destinations.emplace_back(kTerrainDestinationName, sro::math::Vector3(x, heightOnTerrain, y), Destination::Tag::kStart);
    destinations.emplace_back(kTerrainDestinationName, sro::math::Vector3(x, heightOnTerrain, y), Destination::Tag::kGoal);
  }

  // Now, look at every object here
  for (const auto &objectData : regionItem->getNavmeshTriangulation().getObjectDatasForTriangle(*triangleIndex)) {
    const auto &objectInstance = regionItem->getNavmesh().getObjectInstance(objectData.objectInstanceId);
    const auto &objectResource = regionItem->getNavmesh().getObjectResource(objectInstance.objectId);
    const auto transformedPoint = regionItem->getNavmesh().transformPointIntoObjectFrame({static_cast<float>(x), 0.0f, static_cast<float>(y)}, regionId, objectData.objectInstanceId);
    const auto heightOnObject = objectResource.getHeight(transformedPoint, objectData.objectAreaId) + objectInstance.center.y;

    destinations.emplace_back(std::to_string(objectData.objectInstanceId)/* objectResource.name */, sro::math::Vector3(x, heightOnObject, y), Destination::Tag::kStart);
    destinations.emplace_back(std::to_string(objectData.objectInstanceId)/* objectResource.name */, sro::math::Vector3(x, heightOnObject, y), Destination::Tag::kGoal);
  }

  // Create the context menu
  QMenu *menu = new QMenu(this);

  if (!destinations.empty()) {
    // Sort so that higher points are first
    std::stable_sort(destinations.begin(), destinations.end(), [](const auto &a, const auto &b) {
      return a.pos.y > b.pos.y;
    });
    // Sort so that start options are first and goal options are last
    std::stable_sort(destinations.begin(), destinations.end(), [](const auto &a, const auto &b) {
      return static_cast<std::underlying_type_t<Destination::Tag>>(a.tag) < static_cast<std::underlying_type_t<Destination::Tag>>(b.tag);
    });

    bool switchedFromStartToGoal{false};
    for (int i=0; i<destinations.size(); ++i) {
      const auto &destination = destinations[i];
      if (destination.tag == Destination::Tag::kGoal && !switchedFromStartToGoal) {
        // Put a separator between start and goal options
        menu->addSeparator();
        switchedFromStartToGoal = true;
      }

      // Create a context menu item for this coordinate
      QAction *action = new QAction(QString("Set path %1 to %2 %3,%4,%5").arg((destination.tag == Destination::Tag::kStart) ? "start" : "goal").arg(QString::fromStdString(destination.name)).arg(destination.pos.x, 0, 'f', 1).arg(destination.pos.y, 0, 'f', 1).arg(destination.pos.z, 0, 'f', 1), menu);

      if (destination.tag == Destination::Tag::kStart) {
        connect(action, &QAction::triggered, this, [=]() {
          emit setPathStart(destination.pos, regionId);
        });
      } else {
        connect(action, &QAction::triggered, this, [=]() {
          emit setPathGoal(destination.pos, regionId);
        });
      }

      menu->addAction(action);
    }

    menu->addSeparator();
  }

  // Create an action to reset the path
  QAction *resetPathAction = new QAction(QString("Reset path"), menu);
  connect(resetPathAction, &QAction::triggered, this, [=]() {
    emit resetPath();
  });
  menu->addAction(resetPathAction);

  // Display menu asynchronously
  menu->popup(event->globalPos());
}

void NavmeshGraphicsView::mouseMoveEvent(QMouseEvent *event) {
  if (navmeshTriangulation_ == nullptr) {
    // Navmesh triangulation not yet set
    return;
  }
  const auto transformedPos = mapToScene(event->pos());
  const int regionX = transformedPos.x()/1920.0;
  const int regionY = 128 - transformedPos.y()/1920.0 + 1;
  const auto regionId = sro::position_math::worldRegionIdFromSectors(regionX,regionY);
  const float x = fmod(transformedPos.x(),1920.0);
  const float y = 1920.0 - fmod(transformedPos.y(),1920.0);
  const auto absolutePos = navmeshTriangulation_->transformRegionPointIntoAbsolute({x,0.0,y}, regionId);
  emit mouseMoved(absolutePos.x, absolutePos.z, regionX, regionY, regionId, x, y);
  event->ignore();
  QGraphicsView::mouseMoveEvent(event);
}

void NavmeshGraphicsView::showTriangleLabelsSettingToggled(bool isSet) {
  showTriangleLabels_ = isSet;
  updateLabels();
}

void NavmeshGraphicsView::showEdgeLabelsSettingToggled(bool isSet) {
  showEdgeLabels_ = isSet;
  updateLabels();
}

void NavmeshGraphicsView::showVertexLabelsSettingToggled(bool isSet) {
  showVertexLabels_ = isSet;
  updateLabels();
}

void NavmeshGraphicsView::updateLabels() {
  if (navmeshView_->getZoomLevel() >= kMinimumZoomForLabels_) {
    // Zoomed in
    for (auto *item : items()) {
      // Tell each region to add their labels to the scene
      //  Note: It is the region's responsibility to protect against adding labels multiple times
      auto *regionItem = qgraphicsitem_cast<RegionGraphicsItem*>(item);
      if (regionItem != nullptr) {
        if (showTriangleLabels_) {
          regionItem->addTriangleLabels();
        } else {
          regionItem->removeTriangleLabels();
        }
        if (showEdgeLabels_) {
          regionItem->addEdgeLabels();
        } else {
          regionItem->removeEdgeLabels();
        }
        if (showVertexLabels_) {
          regionItem->addVertexLabels();
        } else {
          regionItem->removeVertexLabels();
        }
      }
    }
  } else {
    // Zoomed out
    for (auto *item : items()) {
      // Tell each region to remove their labels from the scene
      //  Note: It is the region's responsibility to protect against removing labels multiple times
      auto *regionItem = qgraphicsitem_cast<RegionGraphicsItem*>(item);
      if (regionItem != nullptr) {
        regionItem->removeTriangleLabels();
        regionItem->removeEdgeLabels();
        regionItem->removeVertexLabels();
      }
    }
  }
}

// =====================================================================================================
// ============================================ NavmeshView ============================================
// =====================================================================================================

void NavmeshView::setNavmeshGraphicsView(NavmeshGraphicsView *navmeshGraphicsView) {
  navmeshGraphicsView_ = navmeshGraphicsView;
}

QGraphicsView* NavmeshView::getView() const {
  return static_cast<QGraphicsView*>(navmeshGraphicsView_);
}

void NavmeshView::zoomIn(double diff) {
  zoomLevel_ += diff;
  setupMatrix();
  if (navmeshGraphicsView_ == nullptr) {
    // NavmeshGraphicsView not yet set up
    return;
  }
  navmeshGraphicsView_->updateLabels();
}

void NavmeshView::zoomOut(double diff) {
  zoomLevel_ -= diff;
  setupMatrix();
  if (navmeshGraphicsView_ == nullptr) {
    // NavmeshGraphicsView not yet set up
    return;
  }
  navmeshGraphicsView_->updateLabels();
}

double NavmeshView::getZoomLevel() const {
  return zoomLevel_;
}

void NavmeshView::setupMatrix() {
  // Originally 0-500
  // scale is 2^([-5,5])
  // qreal scale = qPow(qreal(2), (zoomSlider->value() - 250) / qreal(50));
  qreal scale = qPow(qreal(2), zoomLevel_);

  QTransform matrix;
  matrix.scale(scale, scale);

  if (navmeshGraphicsView_ == nullptr) {
    // NavmeshGraphicsView not yet set up
    return;
  }
  navmeshGraphicsView_->setTransform(matrix);
}
