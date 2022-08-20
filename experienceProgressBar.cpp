#include "experienceProgressBar.hpp"

ExperienceProgressBar::ExperienceProgressBar(QWidget *parent) : QProgressBar(parent) {}

void ExperienceProgressBar::setMaximum(uint64_t maximum) {
  // TODO: Calculate a dynamic scaling factor
  // Max level exp req: 103,622,218,294
  //  = 2^31 * 48.2529
  //  = 2^36.593
  QProgressBar::setMaximum(static_cast<int>(maximum / 100.0));
}

void ExperienceProgressBar::setValue(uint64_t value) {
  // TODO: For some reason, this doesnt update in realtime in the UI
  QProgressBar::setValue(static_cast<int>(value / 100.0));
}

QString ExperienceProgressBar::text() const {
  if (this->maximum() == 0) {
    return QProgressBar::text();
  }
  const double percent = (100.0 * this->value()) / this->maximum();
  return QString(tr("%1%")).arg(percent,0,'f',4);
}
