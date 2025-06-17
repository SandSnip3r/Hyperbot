#include "pairSplitWidget.hpp"
#include "ui_pairSplitWidget.h"

#include <QTableWidgetItem>

PairSplitWidget::PairSplitWidget(QWidget *parent)
    : QWidget(parent), ui_(new Ui::PairSplitWidget) {
  ui_->setupUi(this);
  ui_->teamATable->setColumnCount(1);
  ui_->teamATable->setHorizontalHeaderLabels(QStringList() << tr("Team A"));
  ui_->teamBTable->setColumnCount(1);
  ui_->teamBTable->setHorizontalHeaderLabels(QStringList() << tr("Team B"));

  connect(ui_->teamATable, &QTableWidget::cellClicked, this,
          &PairSplitWidget::onTeamASelectionChanged);
  connect(ui_->teamBTable, &QTableWidget::cellClicked, this,
          &PairSplitWidget::onTeamBSelectionChanged);
}

PairSplitWidget::~PairSplitWidget() { delete ui_; }

void PairSplitWidget::onTeamASelectionChanged(int row, int column) {
  Q_UNUSED(column);
  QTableWidgetItem *itemA = ui_->teamATable->item(row, 0);
  QTableWidgetItem *itemB = ui_->teamBTable->item(row, 0);
  if (itemA && itemB) {
    updateDetailArea(itemA->text(), itemB->text());
  }
}

void PairSplitWidget::onTeamBSelectionChanged(int row, int column) {
  onTeamASelectionChanged(row, column);
}

void PairSplitWidget::updateDetailArea(const QString &teamAName,
                                       const QString &teamBName) {
  ui_->teamALabel->setText(teamAName);
  ui_->teamBLabel->setText(teamBName);
}
