#include "dashboardWidget.hpp"
#include "ui_dashboardwidget.h"

#include "barStyles.hpp"
#include <silkroad_lib/pk2/gameData.hpp>
#include <QProgressBar>
#include <QTreeWidgetItem>
#include <QTreeWidget>
#include <QHeaderView>
#include <QRegularExpression>

DashboardWidget::DashboardWidget(const sro::pk2::GameData &gameData,
                                 QWidget *parent)
    : QWidget(parent), ui(new Ui::DashboardWidget), gameData_(gameData) {
  ui->setupUi(this);
  QStringList headers;
  headers << "Characters" << "HP" << "MP" << "State";
  ui->statusTree->setColumnCount(headers.size());
  ui->statusTree->setHeaderLabels(headers);
  ui->statusTree->header()->setSectionResizeMode(
      ui->statusTree->columnCount() - 1, QHeaderView::Stretch);
  connect(ui->statusTree, &QTreeWidget::itemDoubleClicked, this,
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
  for (auto dialog : detailDialogs_) {
    if (dialog) {
      dialog->close();
    }
  }
  detailDialogs_.clear();
  delete ui;
}

DashboardWidget::PairWidgets &DashboardWidget::ensurePairWidgets(int pairId) {
  if (!pairWidgets_.contains(pairId)) {
    QTreeWidgetItem *item = new QTreeWidgetItem(ui->statusTree);
    item->setExpanded(false);
    QProgressBar *hpBar = new QProgressBar;
    setupHpBar(hpBar);
    hpBar->setRange(0, 0);
    hpBar->setValue(0);
    hpBar->setFormat(QString("0/0"));
    ui->statusTree->setItemWidget(item, 1, hpBar);

    QProgressBar *mpBar = new QProgressBar;
    setupMpBar(mpBar);
    mpBar->setRange(0, 0);
    mpBar->setValue(0);
    mpBar->setFormat(QString("0/0"));
    ui->statusTree->setItemWidget(item, 2, mpBar);

    ui->statusTree->addTopLevelItem(item);
    pairWidgets_.insert(pairId, {item, hpBar, mpBar, "", ""});
  }
  return pairWidgets_[pairId];
}

DashboardWidget::CharacterWidgets &DashboardWidget::ensureCharacterWidgets(const QString &name) {
  if (!characterWidgets_.contains(name)) {
    int pid = pairIdForName(name);
    PairWidgets &pairW = ensurePairWidgets(pid);

    QTreeWidgetItem *item = new QTreeWidgetItem(pairW.item);
    item->setText(0, name);

    QProgressBar *hpBar = new QProgressBar;
    setupHpBar(hpBar);
    hpBar->setRange(0, 0);
    hpBar->setValue(0);
    hpBar->setFormat(QString("0/0"));
    ui->statusTree->setItemWidget(item, 1, hpBar);

    QProgressBar *mpBar = new QProgressBar;
    setupMpBar(mpBar);
    mpBar->setRange(0, 0);
    mpBar->setValue(0);
    mpBar->setFormat(QString("0/0"));
    ui->statusTree->setItemWidget(item, 2, mpBar);

    pairW.item->addChild(item);

    characterWidgets_.insert(name, {item, hpBar, mpBar});
    if (pairW.first.isEmpty()) {
      pairW.first = name;
    } else if (pairW.second.isEmpty()) {
      pairW.second = name;
    }
    updatePairSummary(pid);
  }
  return characterWidgets_[name];
}

void DashboardWidget::onCharacterStatusReceived(QString name, int currentHp,
                                               int maxHp, int currentMp,
                                               int maxMp) {
  CharacterWidgets &widgets = ensureCharacterWidgets(name);

  CharacterData &data = characterData_[name];
  data.currentHp = currentHp;
  data.maxHp = maxHp;
  data.currentMp = currentMp;
  data.maxMp = maxMp;
  widgets.hpBar->setRange(0, maxHp);
  widgets.hpBar->setValue(currentHp);
  widgets.hpBar->setFormat(QString("%1/%2").arg(currentHp).arg(maxHp));

  widgets.mpBar->setRange(0, maxMp);
  widgets.mpBar->setValue(currentMp);
  widgets.mpBar->setFormat(QString("%1/%2").arg(currentMp).arg(maxMp));

  updatePairSummary(pairIdForName(name));
  emit characterDataUpdated(name, data);
}

void DashboardWidget::onActiveStateMachine(QString name, QString stateMachine) {
  CharacterWidgets &widgets = ensureCharacterWidgets(name);
  widgets.item->setText(3, stateMachine);
  characterData_[name].stateMachine = stateMachine;
  updatePairSummary(pairIdForName(name));
  emit characterDataUpdated(name, characterData_.value(name));
}

void DashboardWidget::onSkillCooldowns(QString name, QList<SkillCooldown> cooldowns) {
  characterData_[name].skillCooldowns = cooldowns;
  emit characterDataUpdated(name, characterData_.value(name));
}

void DashboardWidget::clearStatusTable() {
  ui->statusTree->clear();
  characterData_.clear();
  characterWidgets_.clear();
  pairWidgets_.clear();
}

void DashboardWidget::onHyperbotConnected() {
  for (auto dialog : detailDialogs_) {
    if (dialog) {
      dialog->close();
    }
  }
  detailDialogs_.clear();
}

void DashboardWidget::showCharacterDetail(QTreeWidgetItem *item, int column) {
  Q_UNUSED(column);
  if (!item || !item->parent()) {
    return;
  }
  const QString name = item->text(0);
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

int DashboardWidget::pairIdForName(const QString &name) {
  return characterId(name) / 2;
}

void DashboardWidget::updatePairSummary(int pairId) {
  if (!pairWidgets_.contains(pairId)) {
    return;
  }
  PairWidgets &pairW = pairWidgets_[pairId];
  const CharacterData &a = characterData_.value(pairW.first);
  const CharacterData &b = characterData_.value(pairW.second);

  QString names = pairW.first;
  if (!pairW.second.isEmpty()) {
    names += QStringLiteral(" / ") + pairW.second;
  }
  pairW.item->setText(0, names);

  int hpSum = a.currentHp + b.currentHp;
  int hpMax = a.maxHp + b.maxHp;
  pairW.hpBar->setRange(0, hpMax);
  pairW.hpBar->setValue(hpSum);
  pairW.hpBar->setFormat(QString("%1/%2").arg(hpSum).arg(hpMax));

  int mpSum = a.currentMp + b.currentMp;
  int mpMax = a.maxMp + b.maxMp;
  pairW.mpBar->setRange(0, mpMax);
  pairW.mpBar->setValue(mpSum);
  pairW.mpBar->setFormat(QString("%1/%2").arg(mpSum).arg(mpMax));

  QString state = a.stateMachine;
  if (!pairW.second.isEmpty()) {
    if (!state.isEmpty() && !b.stateMachine.isEmpty()) {
      state += " | " + b.stateMachine;
    } else {
      state += b.stateMachine;
    }
  }
  pairW.item->setText(3, state);
}
