#include "interactiveChartView.hpp"

#include <QtCharts/QChart>
#include <QHBoxLayout>
#include <QPainter>
#include <QtMath>

#include <iostream>

InteractiveChartView::InteractiveChartView(QWidget *parent) : QChartView(new QChart(), parent) {
  // Set up the chart.
  chart()->legend()->hide();
  chart()->setTitle("Interactive Chart");

  // Create a default line series and add it to the chart.
  QLineSeries *defaultSeries = new QLineSeries(this);
  series_.append(defaultSeries);
  chart()->addSeries(defaultSeries);

  // Create X axis (e.g., time) and attach it to the series.
  axisX_ = new QValueAxis;
  axisX_->setLabelFormat("%g");
  axisX_->setTitleText("Time");
  chart()->addAxis(axisX_, Qt::AlignBottom);
  defaultSeries->attachAxis(axisX_);

  // Create Y axis (e.g., value) and attach it to the series.
  axisY_ = new QValueAxis;
  axisY_->setLabelFormat("%g");
  axisY_->setTitleText("Value");
  chart()->addAxis(axisY_, Qt::AlignLeft);
  defaultSeries->attachAxis(axisY_);

  // Set an initial default view range.
  // Here we assume a default horizontal width (e.g., 10 units of time) and a vertical range [0,10].
  defaultRect_ = QRectF(0, 0, 10, 10);
  axisX_->setRange(defaultRect_.left(), defaultRect_.right());
  axisY_->setRange(defaultRect_.top(), defaultRect_.bottom());

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

  series_[seriesIndex]->append(point);

  // If auto-follow (x-axis) is enabled, update the x-axis range preserving the current width.
  if (followLatest_) {
    qreal currentWidth = axisX_->max() - axisX_->min();
    axisX_->setRange(point.x() - currentWidth, point.x());
  }

  // Update vertical axis only if the user has not manually set a vertical zoom.
  if (!userYZoom_) {
    updateVerticalAxis();
  }
}

void InteractiveChartView::setHistoricalData(const QVector<QPointF> &data, int seriesIndex) {
  if (seriesIndex < 0 || seriesIndex >= series_.size()) {
    return;
  }

  series_[seriesIndex]->replace(data);

  // Update the x-axis range based on the historical data (if available).
  if (!data.isEmpty()) {
    qreal minX = data.first().x();
    qreal maxX = data.last().x();
    axisX_->setRange(minX, maxX);
  }
  updateVerticalAxis();
}

void InteractiveChartView::resetView() {
  // Reset flags to default auto-follow and auto vertical scaling.
  followLatest_ = true;
  userXZoom_ = false;
  userYZoom_ = false;

  // Reset the horizontal axis to show the default width ending at the latest data point.
  if (!series_.isEmpty() && !series_[0]->points().isEmpty()) {
    qreal latestX = series_[0]->points().last().x();
    axisX_->setRange(latestX - defaultRect_.width(), latestX);
  }
  // Reset the vertical axis to default range.
  axisY_->setRange(defaultRect_.top(), defaultRect_.bottom());
}

void InteractiveChartView::followData() {
  // Shift the current view horizontally to follow live data while preserving the current width.
  if (!series_.isEmpty() && !series_[0]->points().isEmpty()) {
    qreal currentWidth = axisX_->max() - axisX_->min();
    qreal latestX = series_[0]->points().last().x();
    axisX_->setRange(latestX - currentWidth, latestX);
    followLatest_ = true;
    userXZoom_ = false;
  }
}

void InteractiveChartView::wheelEvent(QWheelEvent *event) {
  // Zoom in/out horizontally or vertically based on the wheel event.
  if (event->angleDelta().x() != 0.0) {
    // Horizontal zooming:
    qreal factor = (event->angleDelta().x() > 0) ? 0.9 : 1.1;
    qreal xMin = axisX_->min();
    qreal xMax = axisX_->max();
    qreal center = (xMin + xMax) / 2.0;
    qreal halfRange = (xMax - xMin) / 2.0 * factor;
    axisX_->setRange(center - halfRange, center + halfRange);

    // Check if the new x-range max is nearly equal to the latest data point.
    if (!series_.isEmpty() && !series_[0]->points().isEmpty()) {
      qreal latestX = series_[0]->points().last().x();
      if (qAbs(latestX - axisX_->max()) < 0.001) {
        followLatest_ = true;
        userXZoom_ = false;
      } else {
        // followLatest_ = false;
        userXZoom_ = true;
      }
    }
  } else if (event->angleDelta().y() != 0.0) {
    // Vertical zooming:
    qreal factor = (event->angleDelta().y() > 0) ? 0.9 : 1.1;
    qreal yMin = axisY_->min();
    qreal yMax = axisY_->max();
    qreal center = (yMin + yMax) / 2.0;
    qreal halfRange = (yMax - yMin) / 2.0 * factor;
    axisY_->setRange(center - halfRange, center + halfRange);
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
    if (!series_.isEmpty() && !series_[0]->points().isEmpty()) {
      qreal latestX = series_[0]->points().last().x();
      followLatest_ = (latestX <= axisX_->max() + 0.001);
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
  if (series_.isEmpty() || series_[0]->points().isEmpty()) {
    return;
  }

  qreal xMin = axisX_->min();
  qreal xMax = axisX_->max();
  qreal minY = std::numeric_limits<qreal>::max();
  qreal maxY = std::numeric_limits<qreal>::lowest();

  for (QLineSeries *series : series_) {
    const auto points = series->points();
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
  }
}
