#include "dashboardWidget.hpp"
#include "ui_dashboardwidget.h"

#include "barStyles.hpp"
#include <silkroad_lib/pk2/gameData.hpp>
#include <QProgressBar>
#include <QGridLayout>
#include <QScrollArea>
#include <QRegularExpression>
#include <algorithm>

DashboardWidget::DashboardWidget(const sro::pk2::GameData &gameData,
                                 QWidget *parent)
    : QWidget(parent), ui(new Ui::DashboardWidget), gameData_(gameData) {
  ui->setupUi(this);
  gridContainer_ = ui->scrollAreaWidgetContents;
  gridLayout_ = qobject_cast<QGridLayout *>(gridContainer_->layout());
  if (!gridLayout_) {
    gridLayout_ = new QGridLayout(gridContainer_);
    gridContainer_->setLayout(gridLayout_);
  }
  qRegisterMetaType<CharacterData>("CharacterData");
  qRegisterMetaType<QList<SkillCooldown>>("QList<SkillCooldown>");
}


DashboardWidget::~DashboardWidget() {
  for (auto tile : tiles_) {
    delete tile;
  }
  tiles_.clear();
  delete ui;
}

CharacterTileWidget *DashboardWidget::ensureTileForCharacter(const QString &name) {
  if (tiles_.contains(name)) {
    return tiles_[name];
  }

  CharacterTileWidget *tile = new CharacterTileWidget(gameData_, gridContainer_);
  tile->setCharacterName(name);
  tiles_.insert(name, tile);
  updateGridPositions();

  connect(this, &DashboardWidget::characterDataUpdated, tile,
          &CharacterTileWidget::updateCharacterData);
  return tile;
}

void DashboardWidget::onCharacterStatusReceived(QString name, int currentHp,
                                               int maxHp, int currentMp,
                                               int maxMp) {
  CharacterTileWidget *tile = ensureTileForCharacter(name);

  CharacterData &data = characterData_[name];
  data.currentHp = currentHp;
  data.maxHp = maxHp;
  data.currentMp = currentMp;
  data.maxMp = maxMp;
  tile->updateCharacterData(data);
  emit characterDataUpdated(name, data);
}

void DashboardWidget::onActiveStateMachine(QString name, QString stateMachine) {
  CharacterTileWidget *tile = ensureTileForCharacter(name);
  characterData_[name].stateMachine = stateMachine;
  tile->updateCharacterData(characterData_[name]);
  emit characterDataUpdated(name, characterData_.value(name));
}

void DashboardWidget::onSkillCooldowns(QString name, QList<SkillCooldown> cooldowns) {
  characterData_[name].skillCooldowns = cooldowns;
  CharacterTileWidget *tile = ensureTileForCharacter(name);
  tile->updateCharacterData(characterData_[name]);
  emit characterDataUpdated(name, characterData_.value(name));
}

void DashboardWidget::clearStatusTable() {
  for (auto tile : tiles_) {
    gridLayout_->removeWidget(tile);
    delete tile;
  }
  tiles_.clear();
  characterData_.clear();
}

void DashboardWidget::onHyperbotConnected() {
  for (auto tile : tiles_) {
    tile->updateCharacterData(characterData_.value(tiles_.key(tile)));
  }
}

void DashboardWidget::updateGridPositions() {
  int index = 0;
  for (auto name : tiles_.keys()) {
    CharacterTileWidget *tile = tiles_[name];
    int row = index / columns_;
    int col = index % columns_;
    gridLayout_->addWidget(tile, row, col);
    ++index;
  }
}

