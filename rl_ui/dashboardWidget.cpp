#include "dashboardWidget.hpp"
#include "ui_dashboardwidget.h"

#include "barStyles.hpp"
#include "characterCardWidget.hpp"
#include <silkroad_lib/pk2/gameData.hpp>
#include <QProgressBar>
#include <QRegularExpression>

DashboardWidget::DashboardWidget(const sro::pk2::GameData &gameData,
                                 QWidget *parent)
    : QWidget(parent), ui(new Ui::DashboardWidget), gameData_(gameData) {
  ui->setupUi(this);
  gridLayout_ = ui->gridLayout;
  qRegisterMetaType<CharacterData>("CharacterData");
  qRegisterMetaType<QList<SkillCooldown>>("QList<SkillCooldown>");
}

DashboardWidget::~DashboardWidget() {
  for (auto dialog : detailDialogs_) {
    if (dialog) {
      dialog->close();
    }
  }
  detailDialogs_.clear();
  delete ui;
}

int DashboardWidget::ensureCardForCharacter(const QString &name) {
  if (cards_.contains(name)) {
    return gridLayout_->indexOf(cards_.value(name));
  }

  CharacterCardWidget *card = new CharacterCardWidget(this);
  card->setCharacterName(name);
  int index = gridLayout_->count();
  int columns = 4;
  int row = index / columns;
  int column = index % columns;
  gridLayout_->addWidget(card, row, column);
  cards_.insert(name, card);
  connect(card, &CharacterCardWidget::clicked, this,
          [this, name]() { showCharacterDetail(name); });
  return index;
}

void DashboardWidget::onCharacterStatusReceived(QString name, int currentHp,
                                               int maxHp, int currentMp,
                                               int maxMp) {
  ensureCardForCharacter(name);

  CharacterData &data = characterData_[name];
  data.currentHp = currentHp;
  data.maxHp = maxHp;
  data.currentMp = currentMp;
  data.maxMp = maxMp;

  CharacterCardWidget *card = cards_.value(name);
  if (card) {
    card->updateHpMp(currentHp, maxHp, currentMp, maxMp);
  }

  emit characterDataUpdated(name, data);
}

void DashboardWidget::onActiveStateMachine(QString name, QString stateMachine) {
  ensureCardForCharacter(name);
  characterData_[name].stateMachine = stateMachine;
  CharacterCardWidget *card = cards_.value(name);
  if (card) {
    card->setStateText(stateMachine);
  }
  emit characterDataUpdated(name, characterData_.value(name));
}

void DashboardWidget::onSkillCooldowns(QString name, QList<SkillCooldown> cooldowns) {
  characterData_[name].skillCooldowns = cooldowns;
  emit characterDataUpdated(name, characterData_.value(name));
}

void DashboardWidget::clearStatusTable() {
  QLayoutItem *child;
  while ((child = gridLayout_->takeAt(0)) != nullptr) {
    QWidget *widget = child->widget();
    if (widget) {
      widget->deleteLater();
    }
    delete child;
  }
  cards_.clear();
  characterData_.clear();
}

void DashboardWidget::onHyperbotConnected() {
  for (auto dialog : detailDialogs_) {
    if (dialog) {
      dialog->close();
    }
  }
  detailDialogs_.clear();
  clearStatusTable();
}

void DashboardWidget::showCharacterDetail(QString name) {
  if (detailDialogs_.contains(name)) {
    CharacterDetailDialog *dialog = detailDialogs_.value(name);
    if (dialog) {
      dialog->raise();
      dialog->activateWindow();
    }
    return;
  }
  CharacterDetailDialog *dialog = new CharacterDetailDialog(gameData_, this);
  dialog->setAttribute(Qt::WA_DeleteOnClose);
  detailDialogs_.insert(name, dialog);
  connect(dialog, &QObject::destroyed, this,
          [this, name]() { detailDialogs_.remove(name); });
  dialog->setCharacterName(name);
  dialog->updateCharacterData(characterData_.value(name));
  connect(this, &DashboardWidget::characterDataUpdated, dialog,
          &CharacterDetailDialog::onCharacterDataUpdated);
  dialog->show();
}
