#include "pairSplitViewWidget.hpp"
#include "ui_pairSplitViewWidget.h"

PairSplitViewWidget::PairSplitViewWidget(QWidget *parent)
    : QWidget(parent), ui(new Ui::PairSplitViewWidget) {
  ui->setupUi(this);
}

PairSplitViewWidget::~PairSplitViewWidget() {
  delete ui;
}
