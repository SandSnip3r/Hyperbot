#include "characterGraphicsItem.hpp"

#include <QGraphicsSceneContextMenuEvent>

namespace map {

CharacterGraphicsItem::CharacterGraphicsItem(broadcast::EntityType type) : entityType_(type) {
  precomputeWhatToDraw();
}

void CharacterGraphicsItem::precomputeWhatToDraw() {
  constexpr float kChampionRadiusMultiplier{1.25};
  constexpr float kUniqueRadiusMultiplier{1.5};
  constexpr float kGiantRadiusMultiplier{2};

  // Border is thin
  borderPen_.setWidth(0);

  if (entityType_ == broadcast::EntityType::kMonsterPartyGeneral ||
      entityType_ == broadcast::EntityType::kMonsterPartyChampion ||
      entityType_ == broadcast::EntityType::kMonsterPartyGiant) {
    // Party monsters are drawn with a blue border
    borderPen_.setColor({25,166,233});
  } else {
    // Everything else is drawn with a black border
    borderPen_.setColor({0,0,0});
  }

  // Set radius first
  if (entityType_ == broadcast::EntityType::kMonsterChampion ||
      entityType_ == broadcast::EntityType::kMonsterPartyChampion) {
    shapeRadius_ = kEntityCircleBaseRadius_ * kChampionRadiusMultiplier;
  } else if (entityType_ == broadcast::EntityType::kMonsterUnique) {
    shapeRadius_ = kEntityCircleBaseRadius_ * kUniqueRadiusMultiplier;
  } else if (entityType_ == broadcast::EntityType::kMonsterGiant ||
      entityType_ == broadcast::EntityType::kMonsterPartyGiant) {
    shapeRadius_ = kEntityCircleBaseRadius_ * kGiantRadiusMultiplier;
  } else {
    shapeRadius_ = kEntityCircleBaseRadius_;
  }
  updateRadius(shapeRadius_);

  if (entityType_ == broadcast::EntityType::kMonsterGeneral ||
      entityType_ == broadcast::EntityType::kMonsterChampion ||
      entityType_ == broadcast::EntityType::kMonsterGiant ||
      entityType_ == broadcast::EntityType::kMonsterElite ||
      entityType_ == broadcast::EntityType::kMonsterPartyGeneral ||
      entityType_ == broadcast::EntityType::kMonsterPartyChampion ||
      entityType_ == broadcast::EntityType::kMonsterPartyGiant ||
      entityType_ == broadcast::EntityType::kMonsterUnique) {
    // Drawing a monster, lets draw it as a circle with a black border
    if (entityType_ == broadcast::EntityType::kMonsterUnique) {
      // Purple
      circleGradient_.setColorAt(0, {255,128,255});
      circleGradient_.setColorAt(0.66, {207,27,255});
      circleGradient_.setColorAt(1, {32,0,32});
    } else {
      // Red
      circleGradient_.setColorAt(0, {255,128,128});
      circleGradient_.setColorAt(0.33, {255,0,0});
      circleGradient_.setColorAt(1, {32,0,0});
    }
  } else if (entityType_ == broadcast::EntityType::kNonplayerCharacter) {
    // Blue
    circleGradient_.setColorAt(0, {128,128,255});
    circleGradient_.setColorAt(0.33, {48,96,219});
    circleGradient_.setColorAt(1, {0,0,32});
  } else if (entityType_ == broadcast::EntityType::kPlayerCharacter) {
    // Light green
    circleGradient_.setColorAt(0, {255,128,255});
    circleGradient_.setColorAt(0.33, {157, 199, 0});
    circleGradient_.setColorAt(1, {16,32,0});
  } else if (entityType_ == broadcast::EntityType::kSelf) {
    // Tan
    circleGradient_.setColorAt(0, {255,255,128});
    circleGradient_.setColorAt(0.33, {216,158,0});
    circleGradient_.setColorAt(1, {32,16,0});
  } else {
    // Not entirely sure what we're drawing, lets draw a solid black circle
    fillBrush_.setColor({0,0,0});
    return;
  }

  fillBrush_ = QBrush(circleGradient_);
}

void CharacterGraphicsItem::updateRadius(const float newRadius) {
  prepareGeometryChange();
  zoomedShapeRadius_ = newRadius;
  if (fillBrush_.gradient() != nullptr) {
    circleGradient_.setRadius(zoomedShapeRadius_);
    fillBrush_ = QBrush(circleGradient_);
  }
}

QRectF CharacterGraphicsItem::boundingRect() const {
  return QRectF(-zoomedShapeRadius_-1, -zoomedShapeRadius_-1, zoomedShapeRadius_*2+1, zoomedShapeRadius_*2+1);
}

void CharacterGraphicsItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
  static auto previousM11 = painter->worldTransform().m11();
  if (previousM11 != painter->worldTransform().m11()) {
    previousM11 = painter->worldTransform().m11();
  }
  updateRadius(shapeRadius_ * 1/painter->worldTransform().m11());
  painter->save();
  painter->setPen(borderPen_);
  painter->setBrush(fillBrush_);

  // Draw circle
  painter->drawEllipse(QPointF(0,0), zoomedShapeRadius_, zoomedShapeRadius_);

  // If character is dead, draw an X through it
  if (!alive_) {
    QPen pen(Qt::black);
    constexpr float penWidthMultiplier{1.25};
    const auto penWidth = penWidthMultiplier * (shapeRadius_/kEntityCircleBaseRadius_) / painter->worldTransform().m11();
    pen.setWidthF(penWidth);
    painter->setPen(pen);
    const float x = sqrt(zoomedShapeRadius_*zoomedShapeRadius_/2);
    painter->drawLine(QPointF(-x,-x), QPointF(x,x));
    painter->drawLine(QPointF(-x,x), QPointF(x,-x));
  }
  painter->restore();
}

void CharacterGraphicsItem::setDead() {
  alive_ = false;
  update();
}

} // namespace map