#include "characterDetailDialog.hpp"
#include "ui_characterDetailDialog.h"

CharacterDetailDialog::CharacterDetailDialog(QWidget *parent)
    : QDialog(parent), ui(new Ui::CharacterDetailDialog) {
  ui->setupUi(this);
}

CharacterDetailDialog::~CharacterDetailDialog() {
  delete ui;
}

void CharacterDetailDialog::setCharacterName(const QString &name) {
  setWindowTitle(name);
  ui->nameLabel->setText(name);
}

void CharacterDetailDialog::setCharacterData(const CharacterData &data) {
  ui->hpBar->setRange(0, data.maxHp);
  ui->hpBar->setValue(data.currentHp);
  ui->hpBar->setFormat(QString("%1/%2").arg(data.currentHp).arg(data.maxHp));

  ui->mpBar->setRange(0, data.maxMp);
  ui->mpBar->setValue(data.currentMp);
  ui->mpBar->setFormat(QString("%1/%2").arg(data.currentMp).arg(data.maxMp));

  ui->stateMachineLabel->setText(data.stateMachine);
}
