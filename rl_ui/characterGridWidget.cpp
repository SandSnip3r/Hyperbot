#include "characterGridWidget.hpp"

#include <QProgressBar>
#include <QLabel>

CharacterGridWidget::CharacterGridWidget(const sro::pk2::GameData &gameData,
                                         QWidget *parent)
    : QWidget(parent), gameData_(gameData) {
  gridLayout_ = new QGridLayout(this);
  gridLayout_->setSpacing(4);
  gridLayout_->setContentsMargins(0, 0, 0, 0);
  setLayout(gridLayout_);
}

int CharacterGridWidget::ensureCellForCharacter(const QString &name) {
  if (cells_.contains(name)) {
    return 0;
  }
  CharacterCellWidget *cell = new CharacterCellWidget(gameData_, this);
  cell->setCharacterName(name);
  int index = cells_.size();
  cells_.insert(name, cell);
  int row = index / columns_;
  int col = index % columns_;
  gridLayout_->addWidget(cell, row, col);
  return index;
}

void CharacterGridWidget::onCharacterStatusReceived(QString name, int currentHp,
                                                   int maxHp, int currentMp,
                                                   int maxMp) {
  ensureCellForCharacter(name);
  CharacterData &data = characterData_[name];
  data.currentHp = currentHp;
  data.maxHp = maxHp;
  data.currentMp = currentMp;
  data.maxMp = maxMp;
  cells_[name]->updateCharacterData(data);
  emit characterDataUpdated(name, data);
}

void CharacterGridWidget::onActiveStateMachine(QString name,
                                              QString stateMachine) {
  ensureCellForCharacter(name);
  CharacterData &data = characterData_[name];
  data.stateMachine = stateMachine;
  cells_[name]->updateCharacterData(data);
  emit characterDataUpdated(name, data);
}

void CharacterGridWidget::onSkillCooldowns(QString name,
                                          QList<SkillCooldown> cooldowns) {
  ensureCellForCharacter(name);
  CharacterData &data = characterData_[name];
  data.skillCooldowns = cooldowns;
  cells_[name]->updateCharacterData(data);
  emit characterDataUpdated(name, data);
}

void CharacterGridWidget::clearCharacters() {
  qDeleteAll(cells_);
  cells_.clear();
  characterData_.clear();
}
