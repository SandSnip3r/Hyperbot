#include "pairViewWidget.hpp"
#include "ui_pairviewwidget.h"

#include <algorithm>

PairViewWidget::PairViewWidget(QWidget *parent)
    : QWidget(parent), ui_(new Ui::PairViewWidget) {
  ui_->setupUi(this);
  connect(ui_->teamAList, &QListWidget::currentRowChanged, this,
          &PairViewWidget::onPairSelected);
  connect(ui_->teamBList, &QListWidget::currentRowChanged, this,
          &PairViewWidget::onPairSelected);
}

PairViewWidget::~PairViewWidget() { delete ui_; }

void PairViewWidget::setPairList(const QStringList &teamA,
                                 const QStringList &teamB) {
  teamA_ = teamA;
  teamB_ = teamB;
  ui_->teamAList->clear();
  ui_->teamBList->clear();
  int count = std::min(teamA_.size(), teamB_.size());
  for (int i = 0; i < count; ++i) {
    ui_->teamAList->addItem(teamA_[i]);
    ui_->teamBList->addItem(teamB_[i]);
  }
  if (count > 0) {
    ui_->teamAList->setCurrentRow(0);
  }
}

void PairViewWidget::onPairSelected(int row) {
  ui_->teamAList->setCurrentRow(row);
  ui_->teamBList->setCurrentRow(row);
  ui_->statusALabel->setText(teamA_.value(row));
  ui_->statusBLabel->setText(teamB_.value(row));
}
