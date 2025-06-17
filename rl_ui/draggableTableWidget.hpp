#ifndef DRAGGABLE_TABLE_WIDGET_HPP_
#define DRAGGABLE_TABLE_WIDGET_HPP_

#include <QTableWidget>
#include <QPoint>

class DraggableTableWidget : public QTableWidget {
  Q_OBJECT
public:
  using QTableWidget::QTableWidget;

protected:
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;

private:
  QPoint dragStartPos_;
};

#endif // DRAGGABLE_TABLE_WIDGET_HPP_
