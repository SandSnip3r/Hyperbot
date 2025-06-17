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
  headers << "Character" << "HP" << "MP" << "State";
  ui->statusTree->setColumnCount(headers.size());
  ui->statusTree->setHeaderLabels(headers);
  ui->statusTree->header()->setSectionResizeMode(
      headers.size() - 1, QHeaderView::Stretch);
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

int DashboardWidget::pairIdFromName(const QString &name) const {
  QRegularExpression re("RL_(\\d+)");
  QRegularExpressionMatch match = re.match(name);
  if (match.hasMatch()) {
    return match.captured(1).toInt() / 2;
  }
  return -1;
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

QTreeWidgetItem *DashboardWidget::ensureItemsForCharacter(const QString &name) {
  int pairId = pairIdFromName(name);
  QTreeWidgetItem *parentItem = nullptr;
  if (pairId >= 0) {
    if (!pairInfo_.contains(pairId)) {
      PairInfo info;
      info.item = new QTreeWidgetItem(ui->statusTree);
      info.item->setText(0, QString("Pair %1").arg(pairId));
      QProgressBar *hpBar = new QProgressBar;
      setupHpBar(hpBar);
      ui->statusTree->setItemWidget(info.item, 1, hpBar);
      QProgressBar *mpBar = new QProgressBar;
      setupMpBar(mpBar);
      ui->statusTree->setItemWidget(info.item, 2, mpBar);
      pairInfo_.insert(pairId, info);
    }
    parentItem = pairInfo_[pairId].item;
  }

  // Find existing child item
  if (parentItem) {
    for (int i = 0; i < parentItem->childCount(); ++i) {
      if (parentItem->child(i)->text(0) == name) {
        return parentItem->child(i);
      }
    }
  } else {
    for (int i = 0; i < ui->statusTree->topLevelItemCount(); ++i) {
      QTreeWidgetItem *item = ui->statusTree->topLevelItem(i);
      if (item->text(0) == name) {
        return item;
      }
    }
  }

  QTreeWidgetItem *item = nullptr;
  if (parentItem) {
    item = new QTreeWidgetItem(parentItem);
    parentItem->addChild(item);
  } else {
    item = new QTreeWidgetItem(ui->statusTree);
  }
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
  charWidgets_.insert(name, {hpBar, mpBar});

  if (pairId >= 0) {
    PairInfo &info = pairInfo_[pairId];
    if (info.first.isEmpty()) {
      info.first = name;
    } else if (info.second.isEmpty()) {
      info.second = name;
    }
    QString text = info.first;
    if (!info.second.isEmpty()) {
      text += " / " + info.second;
    }
    info.item->setText(0, text);
    info.widgets.hpBar =
        qobject_cast<QProgressBar *>(ui->statusTree->itemWidget(info.item, 1));
    info.widgets.mpBar =
        qobject_cast<QProgressBar *>(ui->statusTree->itemWidget(info.item, 2));
  }

  return item;
}

void DashboardWidget::onCharacterStatusReceived(QString name, int currentHp,
                                               int maxHp, int currentMp,
                                               int maxMp) {
  QTreeWidgetItem *item = ensureItemsForCharacter(name);

  CharacterData &data = characterData_[name];
  data.currentHp = currentHp;
  data.maxHp = maxHp;
  data.currentMp = currentMp;
  data.maxMp = maxMp;

  ItemWidgets widgets = charWidgets_.value(name);
  if (widgets.hpBar) {
    widgets.hpBar->setRange(0, maxHp);
    widgets.hpBar->setValue(currentHp);
    widgets.hpBar->setFormat(QString("%1/%2").arg(currentHp).arg(maxHp));
  }
  if (widgets.mpBar) {
    widgets.mpBar->setRange(0, maxMp);
    widgets.mpBar->setValue(currentMp);
    widgets.mpBar->setFormat(QString("%1/%2").arg(currentMp).arg(maxMp));
  }

  item->setData(0, Qt::UserRole, name);
  if (!ui->statusTree->itemWidget(item, 3)) {
    ui->statusTree->setItemWidget(item, 3, new QWidget);
  }

  int pairId = pairIdFromName(name);
  if (pairId >= 0 && pairInfo_.contains(pairId)) {
    PairInfo &info = pairInfo_[pairId];
    if (info.widgets.hpBar) {
      const CharacterData &d1 = characterData_.value(info.first);
      const CharacterData &d2 = characterData_.value(info.second);
      int maxHpPair = d1.maxHp + d2.maxHp;
      int curHpPair = d1.currentHp + d2.currentHp;
      info.widgets.hpBar->setRange(0, maxHpPair);
      info.widgets.hpBar->setValue(curHpPair);
      info.widgets.hpBar->setFormat(QString("%1/%2").arg(curHpPair).arg(maxHpPair));
    }
    if (info.widgets.mpBar) {
      const CharacterData &d1 = characterData_.value(info.first);
      const CharacterData &d2 = characterData_.value(info.second);
      int maxMpPair = d1.maxMp + d2.maxMp;
      int curMpPair = d1.currentMp + d2.currentMp;
      info.widgets.mpBar->setRange(0, maxMpPair);
      info.widgets.mpBar->setValue(curMpPair);
      info.widgets.mpBar->setFormat(QString("%1/%2").arg(curMpPair).arg(maxMpPair));
    }
  }

  emit characterDataUpdated(name, data);
}

void DashboardWidget::onActiveStateMachine(QString name, QString stateMachine) {
  QTreeWidgetItem *item = ensureItemsForCharacter(name);
  QWidget *stateWidget = ui->statusTree->itemWidget(item, 3);
  if (!stateWidget) {
    stateWidget = new QWidget;
    ui->statusTree->setItemWidget(item, 3, stateWidget);
  }
  item->setText(3, stateMachine);
  characterData_[name].stateMachine = stateMachine;
  emit characterDataUpdated(name, characterData_.value(name));
}

void DashboardWidget::onSkillCooldowns(QString name, QList<SkillCooldown> cooldowns) {
  characterData_[name].skillCooldowns = cooldowns;
  emit characterDataUpdated(name, characterData_.value(name));
}

void DashboardWidget::clearStatusTable() {
  ui->statusTree->clear();
  characterData_.clear();
  charWidgets_.clear();
  pairInfo_.clear();
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
  if (!item) {
    return;
  }
  QString name = item->data(0, Qt::UserRole).toString();
  if (name.isEmpty()) {
    name = item->text(0);
  }
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
