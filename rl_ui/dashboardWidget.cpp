#include "dashboardWidget.hpp"
#include "ui_dashboardwidget.h"

#include <QGridLayout>
#include <QProgressBar>
#include <QRegularExpression>
#include <QScrollArea>

DashboardWidget::DashboardWidget(const sro::pk2::GameData &gameData,
                                 QWidget *parent)
    : QWidget(parent), ui(new Ui::DashboardWidget), gameData_(gameData) {
  ui->setupUi(this);
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

DashboardWidget::~DashboardWidget() { delete ui; }

int DashboardWidget::ensureCellForCharacter(const QString &name) {
  if (characterCells_.contains(name)) {
    return 0;
  }
  constexpr int kColumns = 4;
  int index = characterCells_.size();
  int row = index / kColumns;
  int col = index % kColumns;
  CharacterCellWidget *cell =
      new CharacterCellWidget(gameData_, ui->scrollAreaWidgetContents);
  cell->setCharacterName(name);
  ui->gridLayout->addWidget(cell, row, col);
  characterCells_.insert(name, cell);
  return 0;
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

  characterCells_[name]->updateCharacterData(data);
  emit characterDataUpdated(name, data);
}

void DashboardWidget::onActiveStateMachine(QString name, QString stateMachine) {
  ensureCellForCharacter(name);
  characterData_[name].stateMachine = stateMachine;
  characterCells_[name]->updateCharacterData(characterData_.value(name));
  emit characterDataUpdated(name, characterData_.value(name));
}

void DashboardWidget::onSkillCooldowns(QString name,
                                       QList<SkillCooldown> cooldowns) {
  ensureCellForCharacter(name);
  characterData_[name].skillCooldowns = cooldowns;
  emit characterDataUpdated(name, characterData_.value(name));
}

void DashboardWidget::clearStatusTable() {
  qDeleteAll(characterCells_);
  characterCells_.clear();
  characterData_.clear();
}

void DashboardWidget::onHyperbotConnected() { clearStatusTable(); }
