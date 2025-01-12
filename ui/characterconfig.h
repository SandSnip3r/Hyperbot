#ifndef CHARACTERCONFIG_H
#define CHARACTERCONFIG_H

#include "eventHandler.hpp"
#include "requester.hpp"

#include "ui-proto/old_config.pb.h"
#include "ui-proto/character_config.pb.h"

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
  void configReceived(proto::old_config::Config config);

private:
  Ui::CharacterConfig *ui;
  EventHandler *eventHandler_{nullptr};
  Requester *requester_{nullptr};
  std::optional<proto::old_config::Config> config_;

  const proto::character_config::CharacterConfig* getCurrentCharacterConfig() const;
  proto::character_config::CharacterConfig* getMutableCurrentCharacterConfig();
  void populateUiFromConfig(const proto::character_config::CharacterConfig &config);
  void updateConfigFromUi();
  void updateAutopotionConfigFromUi(proto::character_config::AutopotionConfig &autopotionConfig);
  void updateTrainingConfigFromUi(proto::character_config::TrainingConfig &trainingConfig);
};

#endif // CHARACTERCONFIG_H
