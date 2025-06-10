#include "dashboardWidget.hpp"
#include "ui_dashboardwidget.h"

#include <QProgressBar>
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QRegularExpression>

DashboardWidget::DashboardWidget(QWidget *parent)
    : QWidget(parent), ui(new Ui::DashboardWidget) {
  ui->setupUi(this);
  QStringList headers;
  headers << "Character" << "HP" << "MP" << "State";
  ui->statusTable->setColumnCount(headers.size());
  ui->statusTable->setHorizontalHeaderLabels(headers);
  ui->statusTable->verticalHeader()->setDefaultSectionSize(20);
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
    hpBar->setStyleSheet(R"(
      QProgressBar {
        border: 1px solid black;
        border-radius: 2px;
        color: white;
        background-color: #131113;
      }
      QProgressBar::chunk {
        background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #630410, stop: 0.5455 #ff3c52, stop: 1 #9c0010);
      }
    )");
    hpBar->setRange(0, 0);
    hpBar->setValue(0);
    hpBar->setFormat(QString("0/0"));
    ui->statusTable->setCellWidget(row, 1, hpBar);

    QProgressBar *mpBar = new QProgressBar;
    mpBar->setStyleSheet(R"(
      QProgressBar {
        border: 1px solid black;
        border-radius: 2px;
        color: white;
        background-color: #131113;
      }
      QProgressBar::chunk {
        background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #101c4a, stop: 0.5455 #4a69ce, stop: 1 #182c73);
      }
    )");
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

  auto *hpBar = qobject_cast<QProgressBar *>(ui->statusTable->cellWidget(row, 1));
  auto *mpBar = qobject_cast<QProgressBar *>(ui->statusTable->cellWidget(row, 2));
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
}

void DashboardWidget::onActiveStateMachine(QString name, QString stateMachine) {
  int row = ensureRowForCharacter(name);
  ui->statusTable->setItem(row, 3, new QTableWidgetItem(stateMachine));
}

void DashboardWidget::clearStatusTable() {
  ui->statusTable->setRowCount(0);
}
