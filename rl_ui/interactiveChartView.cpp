#include "interactiveChartView.hpp"

#include <QtCharts/QChart>
#include <QHBoxLayout>
#include <QPainter>
#include <QtMath>

#include <absl/log/log.h>
#include <array>
#include <algorithm>
#include <functional>
#include <random>

std::mt19937 InteractiveChartView::createRandomEngine() {
  std::random_device rd;
  std::array<int, std::mt19937::state_size> seed_data;
  std::generate_n(seed_data.data(), seed_data.size(), std::ref(rd));
  std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
  return std::mt19937(seq);
}

InteractiveChartView::InteractiveChartView(QWidget *parent)
    : QChartView(new QChart(), parent) {
  // Set up the chart.
  chart()->legend()->hide();
  chart()->setTitle("Interactive Chart");

  // Create a default line series and add it to the chart.
  SeriesData defaultData;
  defaultData.series = new QLineSeries(this);
  series_.append(defaultData);
  chart()->addSeries(defaultData.series);

  // Create X axis (e.g., time) and attach it to the series.
  axisX_ = new QValueAxis;
  axisX_->setLabelFormat("%g");
  axisX_->setTitleText("Time");
  chart()->addAxis(axisX_, Qt::AlignBottom);
  defaultData.series->attachAxis(axisX_);

  // Create Y axis (e.g., value) and attach it to the series.
  axisY_ = new QValueAxis;
  axisY_->setLabelFormat("%.0f");
  axisY_->setTitleText("Value");
  chart()->addAxis(axisY_, Qt::AlignLeft);
  defaultData.series->attachAxis(axisY_);

  // Set an initial default view range.
  // Here we assume a default horizontal width (e.g., 10 units of time) and a vertical range [0,10].
  defaultRect_ = QRectF(0, 0, 10, 10);
  axisX_->setRange(defaultRect_.left(), defaultRect_.right());
  axisY_->setRange(defaultRect_.top(), defaultRect_.bottom());
  axisY_->applyNiceNumbers();

  // Create and position the "Default Zoom" button.
  homeButton_ = new QPushButton("Default Zoom", this);
  homeButton_->setFixedSize(100, 25);
  homeButton_->move(10, 10);
  connect(homeButton_, &QPushButton::clicked, this, &InteractiveChartView::resetView);
  homeButton_->raise();

  // Create and position the "Follow Data" button.
  followButton_ = new QPushButton("Follow Data", this);
  followButton_->setFixedSize(100, 25);
  followButton_->move(10, 40);
  connect(followButton_, &QPushButton::clicked, this, &InteractiveChartView::followData);
  followButton_->raise();

  // Enable antialiasing for smoother rendering.
  setRenderHint(QPainter::Antialiasing);
}

void InteractiveChartView::addDataPoint(const QPointF &point, int seriesIndex) {
  if (seriesIndex < 0 || seriesIndex >= series_.size()) {
    return;
  }

  SeriesData &sd = series_[seriesIndex];
  ++sd.count;

  // Assumes incoming x-values are monotonically increasing.
  if (point.x() < latestX_) {
    LOG(WARNING) << "InteractiveChartView: out-of-order data point";
  }
  latestX_ = point.x();

  if (sd.reservoir.size() < static_cast<int>(kSampleSize)) {
    sd.reservoir.append(point);
  } else {
    std::uniform_int_distribution<qint64> dist(0, sd.count - 1);
    qint64 idx = dist(rng_);
    if (idx < kSampleSize) {
      sd.reservoir.remove(idx);
      sd.reservoir.append(point);
    }
  }

  sd.series->replace(sd.reservoir);

  // If auto-follow (x-axis) is enabled, update the x-axis range preserving the current width.
  if (followLatest_) {
    qreal currentWidth = axisX_->max() - axisX_->min();
    axisX_->setRange(latestX_ - currentWidth, latestX_);
  }

  // Update vertical axis only if the user has not manually set a vertical zoom.
  if (!userYZoom_) {
    updateVerticalAxis();
  }
}


void InteractiveChartView::resetView() {
  // Reset flags to default auto-follow and auto vertical scaling.
  followLatest_ = true;
  userXZoom_ = false;
  userYZoom_ = false;

  // Reset the horizontal axis to show the default width ending at the latest data point.
  if (!series_.isEmpty() && latestX_ != 0) {
    axisX_->setRange(latestX_ - defaultRect_.width(), latestX_);
  }
  // Reset the vertical axis to default range.
  axisY_->setRange(defaultRect_.top(), defaultRect_.bottom());
  axisY_->applyNiceNumbers();
}

void InteractiveChartView::followData() {
  // Shift the current view horizontally to follow live data while preserving the current width.
  if (!series_.isEmpty() && latestX_ != 0) {
    qreal currentWidth = axisX_->max() - axisX_->min();
    axisX_->setRange(latestX_ - currentWidth, latestX_);
    followLatest_ = true;
    userXZoom_ = false;
  }
}

void InteractiveChartView::wheelEvent(QWheelEvent *event) {
  // Zoom in/out horizontally or vertically based on the wheel event.
  QPoint angleDelta = event->angleDelta();
  if (angleDelta.isNull()) {
    angleDelta = event->pixelDelta();
  }
  if (angleDelta.x() != 0) {
    // Horizontal zooming:
    qreal factor = (angleDelta.x() > 0) ? 0.9 : 1.1;
    qreal xMin = axisX_->min();
    qreal xMax = axisX_->max();
    qreal center = (xMin + xMax) / 2.0;
    qreal halfRange = (xMax - xMin) / 2.0 * factor;
    axisX_->setRange(center - halfRange, center + halfRange);

    // Check if the new x-range max is nearly equal to the latest data point.
    if (!series_.isEmpty() && latestX_ != 0) {
      if (qAbs(latestX_ - axisX_->max()) < 0.001) {
        followLatest_ = true;
        userXZoom_ = false;
      } else {
        // followLatest_ = false;
        userXZoom_ = true;
      }
    }
  } else if (angleDelta.y() != 0) {
    // Vertical zooming:
    qreal factor = (angleDelta.y() > 0) ? 0.9 : 1.1;
    qreal yMin = axisY_->min();
    qreal yMax = axisY_->max();
    qreal center = (yMin + yMax) / 2.0;
    qreal halfRange = (yMax - yMin) / 2.0 * factor;
    axisY_->setRange(center - halfRange, center + halfRange);
    axisY_->applyNiceNumbers();
    userYZoom_ = true;
  }
  event->accept();
}

void InteractiveChartView::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::RightButton) {
    // Begin panning when Alt is held.
    panning_ = true;
    lastMousePos_ = event->pos();
  } else if (event->button() == Qt::LeftButton) {
    // Begin rubberband (box) zooming if Alt is not held.
    rubberBandActive_ = true;
    rubberBandRect_.setTopLeft(event->pos());
    rubberBandRect_.setBottomRight(event->pos());
  }
  QChartView::mousePressEvent(event);
}

void InteractiveChartView::mouseMoveEvent(QMouseEvent *event) {
  if (panning_) {
    // Calculate mouse movement delta.
    QPoint delta = event->pos() - lastMousePos_;
    lastMousePos_ = event->pos();

    // Convert pixel delta to chart value delta.
    QPointF deltaValue = chart()->mapToValue(QPoint(0,0)) - chart()->mapToValue(delta);
    axisX_->setRange(axisX_->min() + deltaValue.x(), axisX_->max() + deltaValue.x());
    axisY_->setRange(axisY_->min() + deltaValue.y(), axisY_->max() + deltaValue.y());

    // When panning, mark that the user has manually adjusted both axes.
    userXZoom_ = true;
    userYZoom_ = true;

    // Disable auto-follow if the latest data is not visible.
    if (!series_.isEmpty() && latestX_ != 0) {
      followLatest_ = (latestX_ <= axisX_->max() + 0.001);
    }
  } else if (rubberBandActive_) {
    // Update the rubberband rectangle as the user drags.
    rubberBandRect_.setBottomRight(event->pos());
    viewport()->update();
  }
  QChartView::mouseMoveEvent(event);
}

void InteractiveChartView::mouseReleaseEvent(QMouseEvent *event) {
  if (panning_ && event->button() == Qt::RightButton) {
    panning_ = false;
    axisY_->applyNiceNumbers();
  } else if (rubberBandActive_ && event->button() == Qt::LeftButton) {
    rubberBandActive_ = false;
    // Map the rubberband rectangle to chart coordinates.
    QPointF topLeft = chart()->mapToValue(rubberBandRect_.topLeft());
    QPointF bottomRight = chart()->mapToValue(rubberBandRect_.bottomRight());
    QRectF zoomRect(topLeft, bottomRight);
    zoomRect = zoomRect.normalized();

    // Only apply zoom if the rectangle has a valid (nonzero) area.
    if (zoomRect.width() > 0 && zoomRect.height() > 0) {
      axisX_->setRange(zoomRect.left(), zoomRect.right());
      axisY_->setRange(zoomRect.top(), zoomRect.bottom());
      axisY_->applyNiceNumbers();
      followLatest_ = false;
      userXZoom_ = true;
      userYZoom_ = true;
    }
  }
  QChartView::mouseReleaseEvent(event);
}

void InteractiveChartView::resizeEvent(QResizeEvent *event) {
  QChartView::resizeEvent(event);
  // Reposition the buttons when the widget is resized.
  homeButton_->move(10, 10);
  followButton_->move(10, 40);
}

void InteractiveChartView::paintEvent(QPaintEvent *event) {
  QChartView::paintEvent(event);
  // If rubberband zooming is active, draw the selection rectangle.
  if (rubberBandActive_) {
    QPainter painter(viewport());
    painter.setPen(Qt::DashLine);
    painter.drawRect(rubberBandRect_.normalized());
  }
}

void InteractiveChartView::updateVerticalAxis() {
  // Calculate the min and max y-values for all points that fall within the current x-axis range.
  if (series_.isEmpty() || series_[0].reservoir.isEmpty()) {
    return;
  }

  qreal xMin = axisX_->min();
  qreal xMax = axisX_->max();
  qreal minY = std::numeric_limits<qreal>::max();
  qreal maxY = std::numeric_limits<qreal>::lowest();

  for (const SeriesData &sd : series_) {
    const auto points = sd.series->points();
    for (const QPointF &pt : points) {
      if (pt.x() >= xMin && pt.x() <= xMax) {
        if (pt.y() < minY) {
          minY = pt.y();
        }
        if (pt.y() > maxY) {
          maxY = pt.y();
        }
      }
    }
  }

  if (minY < maxY) {
    axisY_->setRange(minY, maxY);
    axisY_->applyNiceNumbers();
  }
}
