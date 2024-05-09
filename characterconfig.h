#ifndef CHARACTERCONFIG_H
#define CHARACTERCONFIG_H

#include "eventHandler.hpp"
#include "requester.hpp"

#include "proto/config.pb.h"

#include <QGroupBox>

#include <optional>

namespace Ui {
class CharacterConfig;
}

class CharacterConfig : public QGroupBox {
Q_OBJECT
public:
  explicit CharacterConfig(QWidget *parent = nullptr);
  ~CharacterConfig();

  void setEventHandlerAndRequester(EventHandler *eventHandler, Requester *requester);

  void saveAndSendConfig();
public slots:
  void configReceived(proto::config::Config config);

private:
  Ui::CharacterConfig *ui;
  EventHandler *eventHandler_{nullptr};
  Requester *requester_{nullptr};
  std::optional<proto::config::Config> config_;

  const proto::config::CharacterConfig* getCurrentCharacterConfig() const;
  proto::config::CharacterConfig* getMutableCurrentCharacterConfig();
  void populateUiFromConfig(const proto::config::CharacterConfig &config);
  void updateConfigFromUi();
  void updateAutopotionConfigFromUi(proto::config::AutopotionConfig &autopotionConfig);
  void updateTrainingConfigFromUi(proto::config::TrainingConfig &trainingConfig);
};

#endif // CHARACTERCONFIG_H
