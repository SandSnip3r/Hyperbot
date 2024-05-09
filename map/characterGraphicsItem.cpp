#include "characterGraphicsItem.hpp"

#include <QGraphicsSceneContextMenuEvent>

namespace map {

CharacterGraphicsItem::CharacterGraphicsItem(CharacterType characterType, std::optional<proto::entity::MonsterRarity> monsterRarity) : characterType_(characterType), monsterRarity_(monsterRarity) {
  if (characterType == CharacterType::kMonster && !monsterRarity_) {
    throw std::runtime_error("Character type is monster, but dont have monster rarity");
  }
  precomputeWhatToDraw();
}

void CharacterGraphicsItem::precomputeWhatToDraw() {
  constexpr float kChampionRadiusMultiplier{1.25};
  constexpr float kUniqueRadiusMultiplier{1.5};
  constexpr float kGiantRadiusMultiplier{2};

  // Border is thin
  borderPen_.setWidth(0);

  if (monsterRarity_ &&
      (*monsterRarity_ == proto::entity::MonsterRarity::kGeneralParty ||
       *monsterRarity_ == proto::entity::MonsterRarity::kChampionParty ||
       *monsterRarity_ == proto::entity::MonsterRarity::kUniqueParty ||
       *monsterRarity_ == proto::entity::MonsterRarity::kGiantParty ||
       *monsterRarity_ == proto::entity::MonsterRarity::kTitanParty ||
       *monsterRarity_ == proto::entity::MonsterRarity::kEliteParty ||
       *monsterRarity_ == proto::entity::MonsterRarity::kEliteStrongParty ||
       *monsterRarity_ == proto::entity::MonsterRarity::kUnique2Party)) {
    // Party monsters are drawn with a blue border
    borderPen_.setColor({25,166,233});
  } else {
    // Everything else is drawn with a black border
    borderPen_.setColor({0,0,0});
  }

  // Set radius first
  if (monsterRarity_ &&
      (*monsterRarity_ == proto::entity::MonsterRarity::kChampion ||
       *monsterRarity_ == proto::entity::MonsterRarity::kChampionParty)) {
    shapeRadius_ = kEntityCircleBaseRadius_ * kChampionRadiusMultiplier;
  } else if (monsterRarity_ &&
             (*monsterRarity_ == proto::entity::MonsterRarity::kGiant ||
              *monsterRarity_ == proto::entity::MonsterRarity::kGiantParty)) {
    shapeRadius_ = kEntityCircleBaseRadius_ * kGiantRadiusMultiplier;
  } else if (monsterRarity_ &&
             (*monsterRarity_ == proto::entity::MonsterRarity::kUnique ||
              *monsterRarity_ == proto::entity::MonsterRarity::kUnique2 ||
              *monsterRarity_ == proto::entity::MonsterRarity::kUniqueParty ||
              *monsterRarity_ == proto::entity::MonsterRarity::kUnique2Party)) {
    shapeRadius_ = kEntityCircleBaseRadius_ * kUniqueRadiusMultiplier;
  } else {
    shapeRadius_ = kEntityCircleBaseRadius_;
  }
  updateRadius(shapeRadius_);

  if (characterType_ == CharacterType::kMonster) {
    // Drawing a monster, lets draw it as a circle with a black border
    if (*monsterRarity_ == proto::entity::MonsterRarity::kUnique ||
        *monsterRarity_ == proto::entity::MonsterRarity::kUnique2 ||
        *monsterRarity_ == proto::entity::MonsterRarity::kUniqueParty ||
        *monsterRarity_ == proto::entity::MonsterRarity::kUnique2Party) {
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
  } else if (characterType_ == CharacterType::kNonPlayerCharacter) {
    // Blue
    circleGradient_.setColorAt(0, {128,128,255});
    circleGradient_.setColorAt(0.33, {48,96,219});
    circleGradient_.setColorAt(1, {0,0,32});
  } else if (characterType_ == CharacterType::kPlayerCharacter) {
    // Light green
    circleGradient_.setColorAt(0, {255,128,255});
    circleGradient_.setColorAt(0.33, {157, 199, 0});
    circleGradient_.setColorAt(1, {16,32,0});
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