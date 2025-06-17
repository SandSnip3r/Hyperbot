#ifndef STATUS_TABLE_WIDGET_HPP_
#define STATUS_TABLE_WIDGET_HPP_

#include <QTableWidget>
#include <QPoint>

class StatusTableWidget : public QTableWidget {
  Q_OBJECT
public:
  explicit StatusTableWidget(QWidget *parent = nullptr);

signals:
  void characterDropped(const QString &name); // not used maybe

protected:
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;

private:
  QPoint dragStartPos_;
};

#endif // STATUS_TABLE_WIDGET_HPP_
