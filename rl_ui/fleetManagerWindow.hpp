#ifndef FLEET_MANAGER_WINDOW_HPP_
#define FLEET_MANAGER_WINDOW_HPP_

#include "dashboardWidget.hpp"
#include "hyperbot.hpp"

#include <silkroad_lib/pk2/gameData.hpp>

#include <QMainWindow>

class FleetManagerWindow : public QMainWindow {
  Q_OBJECT
public:
  explicit FleetManagerWindow(Hyperbot &hyperbot,
                              const sro::pk2::GameData &gameData,
                              QWidget *parent = nullptr);
  ~FleetManagerWindow();

private slots:
  void onDisconnectedFromHyperbot();

private:
  DashboardWidget *dashboardWidget_{nullptr};
  Hyperbot &hyperbot_;
  const sro::pk2::GameData &gameData_;

  void connectSignals();
};

#endif // FLEET_MANAGER_WINDOW_HPP_
