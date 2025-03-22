#ifndef MAIN_WINDOW_HPP_
#define MAIN_WINDOW_HPP_

#include "config.hpp"
#include "hyperbot.hpp"

#include <QMainWindow>
#include <QStringList>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <QTimer>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  explicit MainWindow(Config &&config, Hyperbot &hyperbot, QWidget *parent = nullptr);
  ~MainWindow();

protected:
  // This is called when this window is shown.
  void showEvent(QShowEvent *event) override;

public slots:
  void onConnectedToHyperbot();
  void onDisconnectedFromHyperbot();
  void onTimerTriggered();

private:
  Ui::MainWindow *ui;
  Config config_;
  Hyperbot &hyperbot_;
  bool connectionWindowShown_{false};
  QMainWindow *connectionWindow_{nullptr};
  QLineSeries *series_;
  QValueAxis *xAxis_;
  QTimer *timer_;

  void connectSignals();
  void showConnectionWindow(const QString &windowTitle);
  void testChart();
};
#endif // MAIN_WINDOW_HPP_
