#include "dashboardWidget.hpp"
#include "ui_dashboardwidget.h"

#include "barStyles.hpp"
#include <QScrollArea>
#include <QRegularExpression>

static int characterId(const QString &name) {
  QRegularExpression re("RL_(\\d+)");
  QRegularExpressionMatch match = re.match(name);
  if (match.hasMatch()) {
    return match.captured(1).toInt();
  }
  return name.toInt();
}

DashboardWidget::DashboardWidget(const sro::pk2::GameData &gameData,
                                 QWidget *parent)
    : QWidget(parent), ui(new Ui::DashboardWidget), gameData_(gameData) {
  ui->setupUi(this);
  gridContainer_ = ui->scrollAreaWidgetContents;
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

QColor DashboardWidget::colorForPair(int pair) const {
  static const QVector<QColor> colors = {Qt::red, Qt::green, Qt::blue,
                                         Qt::magenta, Qt::cyan, Qt::yellow};
  return colors[pair % colors.size()];
}

CharacterCardWidget *DashboardWidget::ensureCardForCharacter(const QString &name) {
  if (cardWidgets_.contains(name)) {
    return cardWidgets_.value(name);
  }

  int id = characterId(name);
  int pair = id / 2;
  CharacterCardWidget *card = new CharacterCardWidget(name, gridContainer_);
  card->setPairColor(colorForPair(pair));
  int index = cardWidgets_.size();
  int row = index / 4;
  int column = index % 4;
  gridLayout_->addWidget(card, row, column);
  connect(card, &CharacterCardWidget::clicked, this, &DashboardWidget::showCharacterDetail);
  cardWidgets_.insert(name, card);
  return card;
}

void DashboardWidget::onCharacterStatusReceived(QString name, int currentHp,
                                                int maxHp, int currentMp,
                                                int maxMp) {
  CharacterCardWidget *card = ensureCardForCharacter(name);
  CharacterData &data = characterData_[name];
  data.currentHp = currentHp;
  data.maxHp = maxHp;
  data.currentMp = currentMp;
  data.maxMp = maxMp;
  card->updateStatus(currentHp, maxHp, currentMp, maxMp);
  emit characterDataUpdated(name, data);
}

void DashboardWidget::onActiveStateMachine(QString name, QString stateMachine) {
  CharacterCardWidget *card = ensureCardForCharacter(name);
  Q_UNUSED(card);
  characterData_[name].stateMachine = stateMachine;
  if (card) {
    card->setState(stateMachine);
  }
  emit characterDataUpdated(name, characterData_.value(name));
}

void DashboardWidget::onSkillCooldowns(QString name, QList<SkillCooldown> cooldowns) {
  characterData_[name].skillCooldowns = cooldowns;
  emit characterDataUpdated(name, characterData_.value(name));
}

void DashboardWidget::clearStatusTable() {
  qDeleteAll(cardWidgets_);
  cardWidgets_.clear();
  characterData_.clear();
}

void DashboardWidget::onHyperbotConnected() {
  for (auto dialog : detailDialogs_) {
    if (dialog) {
      dialog->close();
    }
  }
  detailDialogs_.clear();
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
