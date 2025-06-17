#include "dashboardWidget.hpp"
#include "ui_dashboardwidget.h"

#include "barStyles.hpp"
#include <silkroad_lib/pk2/gameData.hpp>
#include <QProgressBar>
#include <QRegularExpression>
#include <QScrollArea>
#include <QLabel>

DashboardWidget::DashboardWidget(const sro::pk2::GameData &gameData,
                                 QWidget *parent)
    : QWidget(parent), ui(new Ui::DashboardWidget), gameData_(gameData) {
  ui->setupUi(this);
  gridLayout_ = ui->gridLayout;
  qRegisterMetaType<CharacterData>("CharacterData");
  qRegisterMetaType<QList<SkillCooldown>>("QList<SkillCooldown>");
}

static int characterId(const QString &name) {
  QRegularExpression re("RL_(\\d+)");
  QRegularExpressionMatch match = re.match(name);
  if (match.hasMatch()) {
    return match.captured(1).toInt();
  }
  return name.toInt();
}

DashboardWidget::~DashboardWidget() {
  qDeleteAll(cellWidgets_);
  cellWidgets_.clear();
  delete ui;
}

int DashboardWidget::ensureCellForCharacter(const QString &name) {
  if (cellWidgets_.contains(name)) {
    return gridLayout_->indexOf(cellWidgets_.value(name));
  }

  int index = cellWidgets_.size();
  int row = index / 4;
  int column = index % 4;
  CharacterCellWidget *cell = new CharacterCellWidget(this);
  cell->setCharacterName(name);
  gridLayout_->addWidget(cell, row, column);
  cellWidgets_.insert(name, cell);
  connect(cell, &CharacterCellWidget::expandRequested, this,
          &DashboardWidget::onCellExpandRequested);
  return index;
}

void DashboardWidget::onCharacterStatusReceived(QString name, int currentHp,
                                               int maxHp, int currentMp,
                                               int maxMp) {
  ensureCellForCharacter(name);

  CharacterData &data = characterData_[name];
  data.currentHp = currentHp;
  data.maxHp = maxHp;
  data.currentMp = currentMp;
  data.maxMp = maxMp;

  cellWidgets_[name]->updateCharacterData(data);
  emit characterDataUpdated(name, data);
}

void DashboardWidget::onActiveStateMachine(QString name, QString stateMachine) {
  ensureCellForCharacter(name);
  characterData_[name].stateMachine = stateMachine;
  cellWidgets_[name]->updateCharacterData(characterData_.value(name));
  emit characterDataUpdated(name, characterData_.value(name));
}

void DashboardWidget::onSkillCooldowns(QString name, QList<SkillCooldown> cooldowns) {
  characterData_[name].skillCooldowns = cooldowns;
  emit characterDataUpdated(name, characterData_.value(name));
}

void DashboardWidget::clearStatus() {
  qDeleteAll(cellWidgets_);
  cellWidgets_.clear();
  characterData_.clear();
}

void DashboardWidget::onHyperbotConnected() {
  clearStatus();
}

void DashboardWidget::onCellExpandRequested(CharacterCellWidget *cell) {
  Q_UNUSED(cell);
}
