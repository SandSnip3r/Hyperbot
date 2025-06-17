#include "dashboardWidget.hpp"
#include "ui_dashboardwidget.h"

#include "barStyles.hpp"
#include <silkroad_lib/pk2/gameData.hpp>
#include <QProgressBar>
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QTableWidget>
#include <QAbstractItemView>
#include <QRegularExpression>

DashboardWidget::DashboardWidget(const sro::pk2::GameData &gameData,
                                 QWidget *parent)
    : QWidget(parent), ui(new Ui::DashboardWidget), gameData_(gameData) {
  ui->setupUi(this);
  QStringList headers;
  headers << "Character" << "HP" << "MP" << "State";
  ui->statusTable->setColumnCount(headers.size());
  ui->statusTable->setHorizontalHeaderLabels(headers);
  ui->statusTable->verticalHeader()->setDefaultSectionSize(20);
  // Set the last column to stretch to fill the remaining space
  ui->statusTable->horizontalHeader()->setSectionResizeMode(
      ui->statusTable->columnCount() - 1, QHeaderView::Stretch);
  ui->statusTable->setSelectionBehavior(QAbstractItemView::SelectRows);
  ui->statusTable->setDragEnabled(true);
  ui->statusTable->setDragDropMode(QAbstractItemView::DragOnly);
  connect(ui->statusTable, &QTableWidget::cellDoubleClicked, this,
          &DashboardWidget::showCharacterDetail);
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
  delete ui;
}

CharacterData DashboardWidget::getCharacterData(QString name) const {
  return characterData_.value(name);
}

int DashboardWidget::ensureRowForCharacter(const QString &name) {
  int row = -1;
  for (int i = 0; i < ui->statusTable->rowCount(); ++i) {
    QTableWidgetItem *item = ui->statusTable->item(i, 0);
    if (item && item->text() == name) {
      row = i;
      break;
    }
  }

  if (row == -1) {
    int id = characterId(name);
    row = 0;
    while (row < ui->statusTable->rowCount() &&
           characterId(ui->statusTable->item(row, 0)->text()) < id) {
      ++row;
    }
    ui->statusTable->insertRow(row);
    ui->statusTable->setRowHeight(row, 20);
    ui->statusTable->setItem(row, 0, new QTableWidgetItem(name));
    ui->statusTable->setItem(row, 3, new QTableWidgetItem(""));

    QProgressBar *hpBar = new QProgressBar;
    setupHpBar(hpBar);
    hpBar->setRange(0, 0);
    hpBar->setValue(0);
    hpBar->setFormat(QString("0/0"));
    ui->statusTable->setCellWidget(row, 1, hpBar);

    QProgressBar *mpBar = new QProgressBar;
    setupMpBar(mpBar);
    mpBar->setRange(0, 0);
    mpBar->setValue(0);
    mpBar->setFormat(QString("0/0"));
    ui->statusTable->setCellWidget(row, 2, mpBar);
  }

  return row;
}

void DashboardWidget::onCharacterStatusReceived(QString name, int currentHp,
                                               int maxHp, int currentMp,
                                               int maxMp) {
  int row = ensureRowForCharacter(name);

  CharacterData &data = characterData_[name];
  data.currentHp = currentHp;
  data.maxHp = maxHp;
  data.currentMp = currentMp;
  data.maxMp = maxMp;

  QProgressBar *hpBar =
      qobject_cast<QProgressBar *>(ui->statusTable->cellWidget(row, 1));
  QProgressBar *mpBar =
      qobject_cast<QProgressBar *>(ui->statusTable->cellWidget(row, 2));
  if (hpBar) {
    hpBar->setRange(0, maxHp);
    hpBar->setValue(currentHp);
    hpBar->setFormat(QString("%1/%2").arg(currentHp).arg(maxHp));
  }
  if (mpBar) {
    mpBar->setRange(0, maxMp);
    mpBar->setValue(currentMp);
    mpBar->setFormat(QString("%1/%2").arg(currentMp).arg(maxMp));
  }
  if (!ui->statusTable->item(row, 3)) {
    ui->statusTable->setItem(row, 3, new QTableWidgetItem(""));
  }
  emit characterDataUpdated(name, data);
}

void DashboardWidget::onActiveStateMachine(QString name, QString stateMachine) {
  int row = ensureRowForCharacter(name);
  ui->statusTable->setItem(row, 3, new QTableWidgetItem(stateMachine));
  characterData_[name].stateMachine = stateMachine;
  emit characterDataUpdated(name, characterData_.value(name));
}

void DashboardWidget::onSkillCooldowns(QString name, QList<SkillCooldown> cooldowns) {
  characterData_[name].skillCooldowns = cooldowns;
  emit characterDataUpdated(name, characterData_.value(name));
}

void DashboardWidget::clearStatusTable() {
  ui->statusTable->setRowCount(0);
  characterData_.clear();
}

void DashboardWidget::onHyperbotConnected() {}

void DashboardWidget::showCharacterDetail(int row, int column) {
  Q_UNUSED(column);
  QTableWidgetItem *item = ui->statusTable->item(row, 0);
  if (!item) {
    return;
  }
  const QString name = item->text();
  emit characterDetailRequested(name);
}
