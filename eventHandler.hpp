#ifndef EVENT_HANDLER_HPP_
#define EVENT_HANDLER_HPP_

#include "proto/broadcast.pb.h"

#include <silkroad_lib/position.h>

#include <zmq.hpp>

#include <QObject>

#include <mutex>
#include <thread>

class EventHandler : public QObject {
  Q_OBJECT
public:
  EventHandler(zmq::context_t &context);
  ~EventHandler();
  
  void runAsync();
signals:
  void connected();
  void characterHpUpdateChanged(uint32_t currentHp);
  void characterMpUpdateChanged(uint32_t currentMp);
  void characterMaxHpMpUpdateChanged(uint32_t maxHp, uint32_t maxMp);
  void characterLevelUpdate(int32_t level, int64_t expRequired);
  void characterExperienceUpdate(uint64_t currentExperience, uint32_t currentSpExperience);
  void characterSpUpdate(uint32_t skillPoints);
  void characterNameUpdate(const std::string &name);
  void inventoryGoldAmountUpdate(uint64_t goldAmount);
  void characterMovementBeganToDest(sro::Position currentPosition, sro::Position destinationPosition, float speed);
  void characterMovementBeganTowardAngle(sro::Position currentPosition, uint16_t movementAngle, float speed);
  void characterMovementEnded(sro::Position position);
  void regionNameUpdate(const std::string &regionName);
private:
  zmq::context_t &context_;
  std::atomic<bool> run_;
  std::thread thr_;
  void run();
  void handle(const broadcast::BroadcastMessage &message);
};

#endif // EVENT_HANDLER_HPP_