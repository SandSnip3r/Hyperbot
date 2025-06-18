#ifndef MAIN_WINDOW_HPP_
#define MAIN_WINDOW_HPP_

#include "config.hpp"
#include "hyperbot.hpp"
#include "dashboardWidget.hpp"
#include <silkroad_lib/pk2/gameData.hpp>

#include <QMainWindow>
#include <QStringList>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <QTimer>
#include <QTabWidget>
#include <QVector>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  explicit MainWindow(Config &&config, Hyperbot &hyperbot,
                      const sro::pk2::GameData &gameData,
                      QWidget *parent = nullptr);
  ~MainWindow();

protected:
  // This is called when this window is shown.
  void showEvent(QShowEvent *event) override;

public slots:
  void onConnectedToHyperbot();
  void onDisconnectedFromHyperbot();
  void onTimerTriggered();
  void addDashboardTab();
  void closeDashboardTab(int index);

private:
  Ui::MainWindow *ui;
  Config config_;
  Hyperbot &hyperbot_;
  const sro::pk2::GameData &gameData_;
  DashboardWidget *dashboardWidget_{nullptr};
  QTabWidget *dashboardTabs_{nullptr};
  QVector<DashboardWidget *> dashboards_;
  bool connectionWindowShown_{false};
  QMainWindow *connectionWindow_{nullptr};
  QLineSeries *series_;
  QValueAxis *xAxis_;
  QTimer *timer_;
  float minX_{0};
  float maxX_{0};
  float minY_{0};
  float maxY_{0};

  void connectSignals();
  void showConnectionWindow(const QString &windowTitle);
  void testChart();
  void addDataPoint(qreal x, qreal y);
  void connectDashboard(DashboardWidget *widget);
};
#endif // MAIN_WINDOW_HPP_
