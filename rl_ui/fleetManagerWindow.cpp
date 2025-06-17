#include "fleetManagerWindow.hpp"

FleetManagerWindow::FleetManagerWindow(Hyperbot &hyperbot,
                                       const sro::pk2::GameData &gameData,
                                       QWidget *parent)
    : QMainWindow(parent),
      dashboardWidget_(new DashboardWidget(gameData, this)),
      hyperbot_(hyperbot),
      gameData_(gameData) {
  setWindowTitle(tr("Fleet Manager"));
  setCentralWidget(dashboardWidget_);
  connectSignals();
}

FleetManagerWindow::~FleetManagerWindow() = default;

void FleetManagerWindow::connectSignals() {
  connect(&hyperbot_, &Hyperbot::characterStatusReceived, dashboardWidget_,
          &DashboardWidget::onCharacterStatusReceived);
  connect(&hyperbot_, &Hyperbot::activeStateMachineReceived, dashboardWidget_,
          &DashboardWidget::onActiveStateMachine);
  connect(&hyperbot_, &Hyperbot::skillCooldownsReceived, dashboardWidget_,
          &DashboardWidget::onSkillCooldowns);
  connect(&hyperbot_, &Hyperbot::disconnected, this,
          &FleetManagerWindow::onDisconnectedFromHyperbot);
}

void FleetManagerWindow::onDisconnectedFromHyperbot() {
  dashboardWidget_->clearStatusTable();
}
