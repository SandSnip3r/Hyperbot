#include "regionGraphicsItem.hpp"
#include "noScaleLabel.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <random>
#include <iomanip>
#include <set>

#include <QGraphicsScene>

std::mt19937 createRandomEngine() {
  std::random_device rd;
  std::array<int, std::mt19937::state_size> seed_data;
  std::generate_n(seed_data.data(), seed_data.size(), std::ref(rd));
  std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
  return std::mt19937(seq);
}

std::map<sro::navmesh::triangulation::ObjectData, QColor> RegionGraphicsItem::objectColorMap_;

RegionGraphicsItem::RegionGraphicsItem(const sro::navmesh::Navmesh &navmesh, const sro::navmesh::Region &region, const sro::navmesh::triangulation::SingleRegionNavmeshTriangulation &navmeshTriangulation) : navmesh_(navmesh), region_(region), navmeshTriangulation_(navmeshTriangulation) {
}

RegionGraphicsItem::~RegionGraphicsItem() {
  if (!labelsCreated_) {
    // No labels yet, nothing to do
    // TODO: Separate thread could still be creating them
    return;
  }

  // Labels might not belong to the scene, if so, it is our responsibility to delete them
  if (triangleLabelStatus_ == LabelStatus::kAdded) {
    for (auto *label : triangleLabels_) {
      delete label;
    }
    triangleLabels_.clear();
    triangleLabelStatus_ = LabelStatus::kNotAdded;
  }
  if (edgeLabelStatus_ == LabelStatus::kAdded) {
    for (auto *label : edgeLabels_) {
      delete label;
    }
    edgeLabels_.clear();
    edgeLabelStatus_ = LabelStatus::kNotAdded;
  }
  if (vertexLabelStatus_ == LabelStatus::kAdded) {
    for (auto *label : vertexLabels_) {
      delete label;
    }
    vertexLabels_.clear();
    vertexLabelStatus_ = LabelStatus::kNotAdded;
  }
}

QRectF RegionGraphicsItem::boundingRect() const {
  // TODO: Accurately account for pen width too
  return QRectF(0-1, 0-1, 1920+1, 1920+1);
}

void RegionGraphicsItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
  addOrRemoveLabelsIfNeccessary();

  QBrush brush(Qt::GlobalColor::black);
  painter->setBrush(brush);
  painter->setRenderHint(QPainter::Antialiasing, true);

  // TODO: Somehow need to solve for the tiny gap that's shown between regions at certain zoom levels
  QRect target(-1,-1,1921,1921);
  QRect src(0,0,pixmap_.width(),pixmap_.height());
  painter->drawPixmap(target, pixmap_, src);

  drawBlockedTiles(*painter, *option);

  drawObjectColors(*painter, *option);
  drawLinkAreas(*painter, *option);

  // drawVertices(*painter, *option);
  drawEdges(*painter, *option);

  // drawEdgeLinks(*painter, *option);
}

void RegionGraphicsItem::addTriangleLabels() {
  if (triangleLabelStatus_ == LabelStatus::kNotAdded) {
    triangleLabelStatus_ = LabelStatus::kNeedToAdd;
    update();
  } else if (triangleLabelStatus_ == LabelStatus::kNeedToRemove) {
    triangleLabelStatus_ = LabelStatus::kAdded;
  }
}

void RegionGraphicsItem::addEdgeLabels() {
  if (edgeLabelStatus_ == LabelStatus::kNotAdded) {
    edgeLabelStatus_ = LabelStatus::kNeedToAdd;
    update();
  } else if (edgeLabelStatus_ == LabelStatus::kNeedToRemove) {
    edgeLabelStatus_ = LabelStatus::kAdded;
  }
}

void RegionGraphicsItem::addVertexLabels() {
  if (vertexLabelStatus_ == LabelStatus::kNotAdded) {
    vertexLabelStatus_ = LabelStatus::kNeedToAdd;
    update();
  } else if (vertexLabelStatus_ == LabelStatus::kNeedToRemove) {
    vertexLabelStatus_ = LabelStatus::kAdded;
  }
}

void RegionGraphicsItem::removeTriangleLabels() {
  if (triangleLabelStatus_ == LabelStatus::kAdded) {
    triangleLabelStatus_ = LabelStatus::kNeedToRemove;
    update();
  } else if (triangleLabelStatus_ == LabelStatus::kNeedToAdd) {
    triangleLabelStatus_ = LabelStatus::kNotAdded;
  }
}

void RegionGraphicsItem::removeEdgeLabels() {
  if (edgeLabelStatus_ == LabelStatus::kAdded) {
    edgeLabelStatus_ = LabelStatus::kNeedToRemove;
    update();
  } else if (edgeLabelStatus_ == LabelStatus::kNeedToAdd) {
    edgeLabelStatus_ = LabelStatus::kNotAdded;
  }
}

void RegionGraphicsItem::removeVertexLabels() {
  if (vertexLabelStatus_ == LabelStatus::kAdded) {
    vertexLabelStatus_ = LabelStatus::kNeedToRemove;
    update();
  } else if (vertexLabelStatus_ == LabelStatus::kNeedToAdd) {
    vertexLabelStatus_ = LabelStatus::kNotAdded;
  }
}

void RegionGraphicsItem::createLabels() {
  const qreal fontSize{9};
  const QBrush blackBrush(Qt::black);
  const QBrush redBrush(Qt::red);
  const QBrush blueBrush(QColor(0,100,255));
  const auto font = [&fontSize](){
    QFont font("Cascadia Mono");
    font.setPointSizeF(fontSize);
    return font;
  }();

  // Create triangle labels
  const auto triangleCount = navmeshTriangulation_.getTriangleCount();
  triangleLabels_.reserve(triangleCount);
  for (int triangleIndex=0; triangleIndex<triangleCount; ++triangleIndex) {
    const auto [vertexA, vertexB, vertexC] = navmeshTriangulation_.getTriangleVertices(triangleIndex);
    const auto triangleCenter = transformNavmeshCoordinateToQtCoordinate(pathfinder::Vector{(vertexA.x()+vertexB.x()+vertexC.x())/3, (vertexA.y()+vertexB.y()+vertexC.y())/3});
    NoScaleLabel *triangleLabel = new NoScaleLabel(QString::number(triangleIndex));
    triangleLabel->setFont(font);
    triangleLabel->setBrush(blackBrush);
    triangleLabel->setPos(scenePos().x()+triangleCenter.x(), scenePos().y()+triangleCenter.y());
    triangleLabels_.push_back(triangleLabel);
  }

  // Create edge labels
  const auto edgeCount = navmeshTriangulation_.getEdgeCount();
  edgeLabels_.reserve(edgeCount);
  for (int edgeIndex=0; edgeIndex<edgeCount; ++edgeIndex) {
    const auto [vertexA, vertexB] = navmeshTriangulation_.getEdge(edgeIndex);
    const auto edgeCenter = transformNavmeshCoordinateToQtCoordinate(pathfinder::Vector{(vertexA.x()+vertexB.x())/2, (vertexA.y()+vertexB.y())/2});
    NoScaleLabel *edgeLabel = new NoScaleLabel(QString::number(edgeIndex));
    edgeLabel->setFont(font);
    edgeLabel->setBrush(redBrush);
    edgeLabel->setPos(scenePos().x()+edgeCenter.x(), scenePos().y()+edgeCenter.y());
    edgeLabels_.push_back(edgeLabel);
  }

  // Create vertex labels
  const auto vertexCount = navmeshTriangulation_.getVertexCount();
  vertexLabels_.reserve(vertexCount);
  for (int vertexIndex=0; vertexIndex<vertexCount; ++vertexIndex) {
    const auto &vertex = navmeshTriangulation_.getVertex(vertexIndex);
    const auto transformedVertex = transformNavmeshCoordinateToQtCoordinate(vertex);
    NoScaleLabel *vertexLabel = new NoScaleLabel(QString::number(vertexIndex));
    vertexLabel->setFont(font);
    vertexLabel->setBrush(blueBrush);
    vertexLabel->setPos(scenePos().x()+transformedVertex.x(), scenePos().y()+transformedVertex.y());
    vertexLabels_.push_back(vertexLabel);
  }

  labelsCreated_ = true;
}

void RegionGraphicsItem::addOrRemoveLabelsIfNeccessary() {
  if (!labelsCreated_) {
    // No labels yet, nothing to do
    return;
  }

  bool addedOrRemovedSomething{false};
  // Check if we need to add or remove triangle labels
  if (triangleLabelStatus_ == LabelStatus::kNeedToAdd) {
    // Add triangle labels
    auto *ourScene = scene();
    for (auto *label : triangleLabels_) {
      ourScene->addItem(label);
    }
    triangleLabelStatus_ = LabelStatus::kAdded;
    addedOrRemovedSomething = true;
  } else if (triangleLabelStatus_ == LabelStatus::kNeedToRemove) {
    // Remove triangle labels
    auto *ourScene = scene();
    for (auto *label : triangleLabels_) {
      ourScene->removeItem(label);
    }
    triangleLabelStatus_ = LabelStatus::kNotAdded;
    addedOrRemovedSomething = true;
  }

  // Check if we need to add or remove edge labels
  if (edgeLabelStatus_ == LabelStatus::kNeedToAdd) {
    // Add edge labels
    auto *ourScene = scene();
    for (auto *label : edgeLabels_) {
      ourScene->addItem(label);
    }
    edgeLabelStatus_ = LabelStatus::kAdded;
    addedOrRemovedSomething = true;
  } else if (edgeLabelStatus_ == LabelStatus::kNeedToRemove) {
    // Remove edge labels
    auto *ourScene = scene();
    for (auto *label : edgeLabels_) {
      ourScene->removeItem(label);
    }
    edgeLabelStatus_ = LabelStatus::kNotAdded;
    addedOrRemovedSomething = true;
  }

  // Check if we need to add or remove vertex labels
  if (vertexLabelStatus_ == LabelStatus::kNeedToAdd) {
    // Add vertex labels
    auto *ourScene = scene();
    for (auto *label : vertexLabels_) {
      ourScene->addItem(label);
    }
    vertexLabelStatus_ = LabelStatus::kAdded;
    addedOrRemovedSomething = true;
  } else if (vertexLabelStatus_ == LabelStatus::kNeedToRemove) {
    // Remove vertex labels
    auto *ourScene = scene();
    for (auto *label : vertexLabels_) {
      ourScene->removeItem(label);
    }
    vertexLabelStatus_ = LabelStatus::kNotAdded;
    addedOrRemovedSomething = true;
  }

  if (addedOrRemovedSomething) {
    // Update scene for added/removed stuff
    scene()->update();
  }
}

const sro::navmesh::Navmesh& RegionGraphicsItem::getNavmesh() const {
  return navmesh_;
}

const sro::navmesh::Region& RegionGraphicsItem::getRegion() const {
  return region_;
}

const sro::navmesh::triangulation::SingleRegionNavmeshTriangulation& RegionGraphicsItem::getNavmeshTriangulation() const {
  return navmeshTriangulation_;
}

void RegionGraphicsItem::drawBlockedTiles(QPainter &painter, const QStyleOptionGraphicsItem &option) const {
  const qreal levelOfDetail = option.levelOfDetailFromTransform(painter.worldTransform());
  if (levelOfDetail >= 0.1) {
    painter.save();
    QBrush brush(QColor(255,0,0,100));
    painter.setBrush(brush);
    painter.setPen(Qt::NoPen);
    auto tileXYToQuad = [](int row, int col) -> std::tuple<double,double> {
      // minX, minZ/* , maxX, maxZ */
      return {col*20, row*20/* , (col+1)*20, (row+1)*20 */};
    };
    for (int row=0; row<region_.enabledTiles.size(); ++row) {
      const auto &rowOfTiles = region_.enabledTiles[row];
      for (int col=0; col<rowOfTiles.size(); ++col) {
        if (!rowOfTiles[col]) {
          // Disabled tile!
          auto startOfRect = tileXYToQuad(row, col);
          auto endOfRect = tileXYToQuad(row+1, col+1);
          const auto transformedStartVertex = transformNavmeshCoordinateToQtCoordinate(pathfinder::Vector(std::get<0>(startOfRect), std::get<1>(startOfRect)));
          const auto transformedEndVertex = transformNavmeshCoordinateToQtCoordinate(pathfinder::Vector(std::get<0>(endOfRect), std::get<1>(endOfRect)));
          painter.drawRect(QRectF(transformedStartVertex, transformedEndVertex));
        }
      }
    }
    painter.restore();
  }
}

void RegionGraphicsItem::drawObjectColors(QPainter &painter, const QStyleOptionGraphicsItem &option) const {
  auto randomColor = []() {
    static auto eng = createRandomEngine();
    static std::uniform_int_distribution<int> colorDist(0,200);
    return QColor(colorDist(eng), colorDist(eng), colorDist(eng), 75);
  };
  const qreal levelOfDetail = option.levelOfDetailFromTransform(painter.worldTransform());
  if (levelOfDetail >= 0.1) {
    painter.save();
    painter.setPen(Qt::NoPen);
    for (int triangleIndex=0; triangleIndex<navmeshTriangulation_.getTriangleCount(); ++triangleIndex) {
      const auto &objectDatas = navmeshTriangulation_.getObjectDatasForTriangle(triangleIndex);
      for (const auto thisTriangleObjectData : objectDatas) {
        auto objectColorIt = objectColorMap_.find(thisTriangleObjectData);
        if (objectColorIt == objectColorMap_.end()) {
          // This is a new object, create a random color for it
          const auto res = objectColorMap_.emplace(thisTriangleObjectData, randomColor());
          objectColorIt = res.first;
        }
        const auto [vertexA, vertexB, vertexC] = navmeshTriangulation_.getTriangleVertices(triangleIndex);
        const auto transformedVertexA = transformNavmeshCoordinateToQtCoordinate(vertexA);
        const auto transformedVertexB = transformNavmeshCoordinateToQtCoordinate(vertexB);
        const auto transformedVertexC = transformNavmeshCoordinateToQtCoordinate(vertexC);
        QBrush brush{objectColorIt->second};
        QPolygonF triangle;
        triangle.append(transformedVertexA);
        triangle.append(transformedVertexB);
        triangle.append(transformedVertexC);
        painter.setBrush(brush);
        painter.drawPolygon(triangle);
      }
    }
    painter.restore();
  }
}

void RegionGraphicsItem::drawLinkAreas(QPainter &painter, const QStyleOptionGraphicsItem &option) const {
  auto randomColor = []() {
    static auto eng = createRandomEngine();
    static std::uniform_int_distribution<int> colorDist(0,200);
    return QColor(colorDist(eng), colorDist(eng), colorDist(eng), 75);
  };
  const qreal levelOfDetail = option.levelOfDetailFromTransform(painter.worldTransform());
  if (levelOfDetail >= 0.1) {
    painter.save();
    painter.setPen(Qt::NoPen);
    for (int triangleIndex=0; triangleIndex<navmeshTriangulation_.getTriangleCount(); ++triangleIndex) {
      const auto linkId = navmeshTriangulation_.getLinkIdForTriangle(triangleIndex);
      if (linkId.has_value()) {
        auto it = linkColorMap_.find(*linkId);
        if (it == linkColorMap_.end()) {
          // No color for this yet
          const auto res = linkColorMap_.emplace(*linkId, randomColor());
          it = res.first;
        }
        const auto [vertexA, vertexB, vertexC] = navmeshTriangulation_.getTriangleVertices(triangleIndex);
        const auto transformedVertexA = transformNavmeshCoordinateToQtCoordinate(vertexA);
        const auto transformedVertexB = transformNavmeshCoordinateToQtCoordinate(vertexB);
        const auto transformedVertexC = transformNavmeshCoordinateToQtCoordinate(vertexC);
        QBrush brush{it->second};
        QPolygonF triangle;
        triangle.append(transformedVertexA);
        triangle.append(transformedVertexB);
        triangle.append(transformedVertexC);
        painter.setBrush(brush);
        painter.drawPolygon(triangle);
      }
    }
    painter.restore();
  }
}

void RegionGraphicsItem::drawVertices(QPainter &painter, const QStyleOptionGraphicsItem &option) const {
  const qreal levelOfDetail = option.levelOfDetailFromTransform(painter.worldTransform());
  if (levelOfDetail < 2) {
    // Don't draw vertices when we're zoomed out
    return;
  }
  painter.save();
  QPen pen;
  pen.setWidth(0);
  painter.setPen(pen);
  painter.setBrush(QBrush(Qt::black));
  const double kPointRadius = 2 * 1/painter.worldTransform().m11();
  for (std::size_t vertexIndex=0; vertexIndex<navmeshTriangulation_.getVertexCount(); ++vertexIndex) {
    const auto &vertex = navmeshTriangulation_.getVertex(static_cast<pathfinder::navmesh::TriangleLibNavmesh::IndexType>(vertexIndex));
    const auto transformedVertex = transformNavmeshCoordinateToQtCoordinate(vertex);
    painter.drawEllipse(transformedVertex, kPointRadius, kPointRadius);
  }
  painter.restore();
}

void RegionGraphicsItem::drawEdges(QPainter &painter, const QStyleOptionGraphicsItem &option) const {
  constexpr bool kDisplayGlobalEdges{false};
  constexpr bool kDisplayNonConstraintEdges{false};
  constexpr double kMinimumLevelOfDetailForNonConstraintEdges{0.35};
  const qreal levelOfDetail = option.levelOfDetailFromTransform(painter.worldTransform());
  // std::cout << "Level of detail " << levelOfDetail << std::endl;
  painter.save();
  QPen edgePen;
  qreal penWidth = 0;
  if (levelOfDetail < 0.2) {
    penWidth = 4;
  }
  edgePen.setWidth(penWidth);
  for (std::size_t edgeIndex=0; edgeIndex<navmeshTriangulation_.getEdgeCount(); ++edgeIndex) {
    const int marker = navmeshTriangulation_.getEdgeMarker(static_cast<pathfinder::navmesh::TriangleLibNavmesh::IndexType>(edgeIndex));
    if (marker <= 1 && (!kDisplayNonConstraintEdges || levelOfDetail <= kMinimumLevelOfDetailForNonConstraintEdges)) {
      // Do not display non-input edges
      continue;
    }
    const auto &[vertexA, vertexB] = navmeshTriangulation_.getEdge(static_cast<pathfinder::navmesh::TriangleLibNavmesh::IndexType>(edgeIndex));
    const auto transformedVertexA = transformNavmeshCoordinateToQtCoordinate(vertexA);
    const auto transformedVertexB = transformNavmeshCoordinateToQtCoordinate(vertexB);
    bool skip{false};//{marker > 1};
    if (marker > 1) {
      const auto &edgeConstraintData = navmeshTriangulation_.getEdgeConstraintData(marker);
      for (const auto &constraint : edgeConstraintData) {
        if (levelOfDetail <= 0.4 && constraint.forObject() && constraint.is(sro::navmesh::triangulation::EdgeConstraintFlag::kInternal)) {
          // Skip object internal edges when zoomed out very far
          skip = true;
          break;
        }
        if constexpr (!kDisplayGlobalEdges) {
          if (!constraint.forObject() && constraint.is(sro::navmesh::triangulation::EdgeConstraintFlag::kGlobal)) {
            // Skip region boundaries
            skip = true;
            break;
          }
        }
      }
    }
    if (!skip) {
      edgePen.setColor(getColorForEdgeMarker(marker));
      painter.setPen(edgePen);
      painter.drawLine(transformedVertexA, transformedVertexB);
    }
  }
  painter.restore();
}

void RegionGraphicsItem::drawEdgeLinks(QPainter &painter, const QStyleOptionGraphicsItem &option) const {
  using StupleTuple = std::tuple<uint32_t, uint32_t, uint16_t, uint16_t>;
  std::set<StupleTuple> drawn;
  auto ordered = [](const auto objId1, const auto objId2, const auto edgeId1, const auto edgeId2) -> StupleTuple {
    if (objId1 < objId2) {
      return {objId1,objId2,edgeId1,edgeId2};
    } else {
      return {objId2,objId1,edgeId2,edgeId1};
    }
  };
  const qreal levelOfDetail = option.levelOfDetailFromTransform(painter.worldTransform());
  const float circleRadius = 5 * 1/levelOfDetail;
  painter.save();
  painter.setPen(Qt::NoPen);
  painter.setBrush(QBrush(QColor(255,0,255,50)));
  for (const auto objectInstanceId : region_.objectInstanceIds) {
    const auto &objectInstance = navmesh_.getObjectInstance(objectInstanceId);
    const auto transformedObjectResource = navmesh_.getTransformedObjectResourceForRegion(objectInstanceId, region_.id);
    for (const auto &link : objectInstance.globalEdgeLinks) {
      if (link.linkedObjGlobalId == -1 || link.edgeId == -1) {
        // Dont have data for this
        continue;
      }
      if (drawn.find(ordered(objectInstanceId, link.linkedObjGlobalId, link.edgeId, link.linkedObjEdgeId)) != drawn.end()) {
        // Already drawn, skip
        continue;
      }
      const auto transformedOtherObjectResource = navmesh_.getTransformedObjectResourceForRegion(link.linkedObjGlobalId, region_.id);
      const auto &ourEdge = transformedObjectResource.outlineEdges.at(link.edgeId);
      const auto &otherObjectEdge = transformedOtherObjectResource.outlineEdges.at(link.linkedObjEdgeId);
      const auto &ourEdgeVertexA = transformedObjectResource.vertices.at(ourEdge.srcVertex);
      const auto &ourEdgeVertexB = transformedObjectResource.vertices.at(ourEdge.destVertex);
      const auto &otherEdgeVertexA = transformedOtherObjectResource.vertices.at(otherObjectEdge.srcVertex);
      const auto &otherEdgeVertexB = transformedOtherObjectResource.vertices.at(otherObjectEdge.destVertex);
      const auto tranformedOurEdgeCenterVertex = transformNavmeshCoordinateToQtCoordinate({ourEdgeVertexA.x+(ourEdgeVertexB.x-ourEdgeVertexA.x)/2, ourEdgeVertexA.z+(ourEdgeVertexB.z-ourEdgeVertexA.z)/2});
      const auto tranformedOtherEdgeCenterVertex = transformNavmeshCoordinateToQtCoordinate({otherEdgeVertexA.x+(otherEdgeVertexB.x-otherEdgeVertexA.x)/2, otherEdgeVertexA.z+(otherEdgeVertexB.z-otherEdgeVertexA.z)/2});
      painter.drawEllipse(tranformedOurEdgeCenterVertex, circleRadius, circleRadius);
      painter.drawEllipse(tranformedOtherEdgeCenterVertex, circleRadius, circleRadius);
      painter.save();
      QPen pen({QColor(255,0,255)});
      pen.setWidth(0);
      painter.setPen(pen);
      painter.drawLine(tranformedOurEdgeCenterVertex, tranformedOtherEdgeCenterVertex);
      painter.restore();
      drawn.insert(ordered(objectInstanceId, link.linkedObjGlobalId, link.edgeId, link.linkedObjEdgeId));
    }
  }
  painter.restore();
}

QColor RegionGraphicsItem::getColorForEdgeMarker(const int marker) const {
  if (marker == 0) {
    // Non-constraint edge
    return QColor{150,255,150};
  } else if (marker == 1) {
    // Boundary
    return QColor{100,100,100};
  } else {
    // User-defined constraint data
    std::optional<QColor> colorToReturn;
    auto checkAndSet = [&colorToReturn](const QColor &color) {
      if (colorToReturn.has_value()) {
        if (*colorToReturn != color) {
          throw std::runtime_error("Trying to overwrite a color with a different color");
        }
      }
      colorToReturn = color;
    };
    const auto &constraintData = navmeshTriangulation_.getEdgeConstraintData(marker);
    for (const auto &constraint : constraintData) {
      if (constraint.forTerrain()) {
        if (constraint.is(sro::navmesh::triangulation::EdgeConstraintFlag::kBlocking)) {
          // Blocking terrain edge
          checkAndSet(Qt::GlobalColor::red);
        } else {
          // Non-blocking terrain edge (probably only global)
          checkAndSet(Qt::GlobalColor::green);
        }
      } else if (constraint.forObject()) {
        // For object
        if (constraint.is(sro::navmesh::triangulation::EdgeConstraintFlag::kBlocking)) {
          // Blocking edge of object
          checkAndSet(QColor(155,0,0));
        } else if (constraint.is(sro::navmesh::triangulation::EdgeConstraintFlag::kBridge)) {
          // Bridge edge of object
          checkAndSet(Qt::GlobalColor::blue);
        } else {
          // Must be non-blocking external edge of the object?
          checkAndSet(QColor(255,0,255));
        }
      }
    }
    if (colorToReturn.has_value()) {
      return *colorToReturn;
    }
  }
  throw std::runtime_error("Unhandled marker");
}

QPointF RegionGraphicsItem::transformNavmeshCoordinateToQtCoordinate(const pathfinder::Vector &vertex) const {
  return {vertex.x(), 1920-vertex.y()};
}

void RegionGraphicsItem::setPixmap(const QPixmap &pixmap) {
  pixmap_ = pixmap;
}
