#include "dashboardWidget.hpp"
#include "ui_dashboardwidget.h"

#include <QProgressBar>
#include <QTableWidgetItem>

DashboardWidget::DashboardWidget(QWidget *parent) : QWidget(parent), ui(new Ui::DashboardWidget) {
  ui->setupUi(this);
  ui->statusTable->setColumnCount(3);
  QStringList headers;
  headers << "Character" << "HP" << "MP";
  ui->statusTable->setHorizontalHeaderLabels(headers);
}

DashboardWidget::~DashboardWidget() {
  delete ui;
}

void DashboardWidget::onCharacterStatusReceived(QString name, int currentHp,
                                               int maxHp, int currentMp,
                                               int maxMp) {
  int row = -1;
  for (int i = 0; i < ui->statusTable->rowCount(); ++i) {
    QTableWidgetItem *item = ui->statusTable->item(i, 0);
    if (item && item->text() == name) {
      row = i;
      break;
    }
  }

  if (row == -1) {
    row = ui->statusTable->rowCount();
    ui->statusTable->insertRow(row);
    ui->statusTable->setItem(row, 0, new QTableWidgetItem(name));

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
    ui->statusTable->setCellWidget(row, 2, mpBar);
  }

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
}
