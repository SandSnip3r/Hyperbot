#ifndef EXPERIENCE_PROGRESS_BAR_HPP_
#define EXPERIENCE_PROGRESS_BAR_HPP_

#include <QProgressBar>

#include <cstdint>

class ExperienceProgressBar : public QProgressBar {
public:
  ExperienceProgressBar(QWidget *parent=nullptr);
  void setMaximum(uint64_t maximum);
  void setValue(uint64_t value);
  virtual QString text() const override;
};

#endif // EXPERIENCE_PROGRESS_BAR_HPP_
