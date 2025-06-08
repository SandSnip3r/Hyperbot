#ifndef INTERACTIVE_CHART_VIEW_HPP_
#define INTERACTIVE_CHART_VIEW_HPP_

#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <QPushButton>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QRectF>
#include <QVector>

#include <limits>
#include <random>

class InteractiveChartView : public QChartView {
  Q_OBJECT
public:
  explicit InteractiveChartView(QWidget *parent = nullptr);

  /// Add a new data point (streaming data). By default the data is appended to series 0.
  void addDataPoint(const QPointF &point, int seriesIndex = 0);


public slots:
  /// Reset the view to the default scrolling and zoom settings.
  void resetView();

  /// Shift the view to follow live data while preserving the current horizontal zoom width.
  void followData();

protected:
  // Overridden event handlers for zooming, panning, and rubberband (box) zooming.
  void wheelEvent(QWheelEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;
  void paintEvent(QPaintEvent *event) override;

private:
  /// Container for one series and its sampling data.
  struct SeriesData {
    QLineSeries *series{nullptr};
    QVector<QPointF> reservoir;
    qint64 count{0};
  };

  /// Recalculates and updates the vertical (y) axis range based on visible data.
  void updateVerticalAxis();

  /// Creates a seeded random engine for reservoir sampling.
  static std::mt19937 createRandomEngine();

  // Container for one or more data series
  QVector<SeriesData> series_;

  // Reservoir sampling configuration
  static constexpr int kSampleSize = 4096;
  qreal latestX_{0};  ///< Latest observed x-value
  std::mt19937 rng_{createRandomEngine()};

  // Axes for the chart
  QValueAxis *axisX_;
  QValueAxis *axisY_;

  // Flags to control auto-follow and user manual zoom.
  bool followLatest_{true};   // True if x-axis should follow the latest data.
  bool userXZoom_{false};      // True if the user has manually adjusted the x-axis zoom.
  bool userYZoom_{false};      // True if the user has manually adjusted the y-axis zoom.

  // Panning state
  bool panning_{false};        // True when user is panning with Alt+drag.
  QPoint lastMousePos_; // For panning delta calculation.

  // Rubberband (box) zooming state.
  bool rubberBandActive_{false};
  QRect rubberBandRect_;

  // Default view range (for resetting the view).
  QRectF defaultRect_;

  // Buttons for resetting zoom and following data.
  QPushButton *homeButton_;
  QPushButton *followButton_;
};

#endif // INTERACTIVE_CHART_VIEW_HPP_
