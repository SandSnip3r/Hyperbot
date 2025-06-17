#include "cardDashboardWidget.hpp"
#include "barStyles.hpp"

#include <QDockWidget>
#include <QVBoxLayout>
#include <QMainWindow>

CardDashboardWidget::CardDashboardWidget(const sro::pk2::GameData &gameData,
                                         QWidget *parent)
    : QWidget(parent),
      scrollArea_(new QScrollArea(this)),
      container_(new QWidget(scrollArea_)),
      gridLayout_(new QGridLayout(container_)),
      gameData_(gameData) {
  gridLayout_->setSpacing(4);
  container_->setLayout(gridLayout_);
  scrollArea_->setWidgetResizable(true);
  scrollArea_->setWidget(container_);

  QVBoxLayout *layout = new QVBoxLayout(this);
  layout->addWidget(scrollArea_);
  setLayout(layout);
}

CardDashboardWidget::~CardDashboardWidget() {
  for (auto dock : dockMap_) {
    if (dock) {
      dock->close();
    }
  }
}

CharacterCardWidget *CardDashboardWidget::ensureCard(const QString &name) {
  if (cardMap_.contains(name)) {
    return cardMap_[name];
  }
  int index = cardMap_.size();
  int columns = 4;
  int row = index / columns;
  int column = index % columns;

  CharacterCardWidget *card = new CharacterCardWidget(container_);
  card->setCharacterName(name);
  connect(card, &CharacterCardWidget::cardClicked, this,
          &CardDashboardWidget::showCharacterDetail);
  gridLayout_->addWidget(card, row, column);
  cardMap_.insert(name, card);
  return card;
}

void CardDashboardWidget::onCharacterStatusReceived(QString name, int currentHp,
                                                    int maxHp, int currentMp,
                                                    int maxMp) {
  CharacterCardWidget *card = ensureCard(name);
  CharacterData &data = characterData_[name];
  data.currentHp = currentHp;
  data.maxHp = maxHp;
  data.currentMp = currentMp;
  data.maxMp = maxMp;
  card->updateCharacterData(data);
  emit characterDataUpdated(name, data);
}

void CardDashboardWidget::onActiveStateMachine(QString name,
                                               QString stateMachine) {
  CharacterCardWidget *card = ensureCard(name);
  characterData_[name].stateMachine = stateMachine;
  card->updateCharacterData(characterData_[name]);
  emit characterDataUpdated(name, characterData_[name]);
}

void CardDashboardWidget::onSkillCooldowns(QString name,
                                           QList<SkillCooldown> cooldowns) {
  characterData_[name].skillCooldowns = cooldowns;
  emit characterDataUpdated(name, characterData_[name]);
}

void CardDashboardWidget::clearStatus() {
  qDeleteAll(cardMap_);
  cardMap_.clear();
  characterData_.clear();
}

void CardDashboardWidget::onHyperbotConnected() {
  for (auto dock : dockMap_) {
    if (dock) {
      dock->close();
    }
  }
  dockMap_.clear();
}

void CardDashboardWidget::showCharacterDetail(QString name) {
  if (dockMap_.contains(name)) {
    QDockWidget *dock = dockMap_[name];
    dock->raise();
    dock->activateWindow();
    return;
  }
  CharacterDetailDialog *dialog = new CharacterDetailDialog(gameData_);
  dialog->setCharacterName(name);
  dialog->updateCharacterData(characterData_[name]);
  connect(this, &CardDashboardWidget::characterDataUpdated, dialog,
          &CharacterDetailDialog::onCharacterDataUpdated);

  QDockWidget *dock = new QDockWidget(name, qobject_cast<QMainWindow *>(window()));
  dock->setWidget(dialog);
  dock->setAttribute(Qt::WA_DeleteOnClose);
  connect(dock, &QObject::destroyed, this,
          [this, name]() { dockMap_.remove(name); });
  qobject_cast<QMainWindow *>(window())
      ->addDockWidget(Qt::RightDockWidgetArea, dock);
  dock->show();
  dockMap_.insert(name, dock);
}
