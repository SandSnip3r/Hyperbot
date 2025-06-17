#include "pairViewWidget.hpp"
#include "ui_pairViewWidget.h"

#include <QTableWidgetItem>

PairViewWidget::PairViewWidget(const sro::pk2::GameData &gameData,
                               QWidget *parent)
    : QWidget(parent), ui_(new Ui::PairViewWidget), gameData_(gameData) {
  ui_->setupUi(this);
  connect(ui_->pairTable, &QTableWidget::currentRowChanged, this,
          &PairViewWidget::onPairSelectionChanged);
}

PairViewWidget::~PairViewWidget() { delete ui_; }

void PairViewWidget::ensureRowCount(int row) {
  while (ui_->pairTable->rowCount() <= row) {
    ui_->pairTable->insertRow(ui_->pairTable->rowCount());
  }
}

void PairViewWidget::onCharacterStatusReceived(QString name, int currentHp,
                                               int maxHp, int currentMp,
                                               int maxMp) {
  if (!teamA_.contains(name) && !teamB_.contains(name)) {
    if (teamA_.size() <= teamB_.size()) {
      teamA_.push_back(name);
      ensureRowCount(teamA_.size() - 1);
      ui_->pairTable->setItem(teamA_.size() - 1, 0,
                              new QTableWidgetItem(name));
    } else {
      teamB_.push_back(name);
      ensureRowCount(teamB_.size() - 1);
      ui_->pairTable->setItem(teamB_.size() - 1, 1,
                              new QTableWidgetItem(name));
    }
  }
  CharacterData &data = characterData_[name];
  data.currentHp = currentHp;
  data.maxHp = maxHp;
  data.currentMp = currentMp;
  data.maxMp = maxMp;
  updateDetailViews();
}

void PairViewWidget::onActiveStateMachine(QString name, QString stateMachine) {
  characterData_[name].stateMachine = stateMachine;
  updateDetailViews();
}

void PairViewWidget::onSkillCooldowns(QString name,
                                      QList<SkillCooldown> cooldowns) {
  characterData_[name].skillCooldowns = cooldowns;
  updateDetailViews();
}

void PairViewWidget::onHyperbotConnected() {
  clearData();
}

void PairViewWidget::clearData() {
  ui_->pairTable->setRowCount(0);
  teamA_.clear();
  teamB_.clear();
  characterData_.clear();
  selectedRow_ = -1;
  updateDetailViews();
}

void PairViewWidget::onPairSelectionChanged(int currentRow) {
  selectedRow_ = currentRow;
  updateDetailViews();
}

void PairViewWidget::updateDetailViews() {
  if (selectedRow_ < 0) {
    ui_->teamALabel->setText(tr("Team A"));
    ui_->teamBLabel->setText(tr("Team B"));
    return;
  }
  if (selectedRow_ < teamA_.size()) {
    const QString nameA = teamA_[selectedRow_];
    const CharacterData &data = characterData_.value(nameA);
    ui_->teamALabel->setText(
        QString("%1 HP:%2/%3 MP:%4/%5 State:%6")
            .arg(nameA)
            .arg(data.currentHp)
            .arg(data.maxHp)
            .arg(data.currentMp)
            .arg(data.maxMp)
            .arg(data.stateMachine));
  } else {
    ui_->teamALabel->setText(tr("Team A"));
  }
  if (selectedRow_ < teamB_.size()) {
    const QString nameB = teamB_[selectedRow_];
    const CharacterData &data = characterData_.value(nameB);
    ui_->teamBLabel->setText(
        QString("%1 HP:%2/%3 MP:%4/%5 State:%6")
            .arg(nameB)
            .arg(data.currentHp)
            .arg(data.maxHp)
            .arg(data.currentMp)
            .arg(data.maxMp)
            .arg(data.stateMachine));
  } else {
    ui_->teamBLabel->setText(tr("Team B"));
  }
}

