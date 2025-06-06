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

void DashboardWidget::onCharacterStatusListReceived(QStringList statusList) {
  ui->statusTable->clearContents();
  ui->statusTable->setRowCount(statusList.size());
  for (int i = 0; i < statusList.size(); ++i) {
    const QString &entry = statusList.at(i);
    const QStringList parts = entry.split(',');
    QString name = parts.value(0).trimmed();
    int currentHp = parts.value(1).trimmed().toInt();
    int maxHp = parts.value(2).trimmed().toInt();
    int currentMp = parts.value(3).trimmed().toInt();
    int maxMp = parts.value(4).trimmed().toInt();

    ui->statusTable->setItem(i, 0, new QTableWidgetItem(name));

    QProgressBar *hpBar = new QProgressBar;
    hpBar->setRange(0, maxHp);
    hpBar->setValue(currentHp);
    hpBar->setFormat(QString("%1/%2").arg(currentHp).arg(maxHp));
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
    ui->statusTable->setCellWidget(i, 1, hpBar);

    QProgressBar *mpBar = new QProgressBar;
    mpBar->setRange(0, maxMp);
    mpBar->setValue(currentMp);
    mpBar->setFormat(QString("%1/%2").arg(currentMp).arg(maxMp));
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
    ui->statusTable->setCellWidget(i, 2, mpBar);
  }
}
