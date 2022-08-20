#ifndef EVENT_HANDLER_HPP_
#define EVENT_HANDLER_HPP_

#include "proto/broadcast.pb.h"

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
  void message1Received(const std::string &str);
  void message2Received(const int32_t val);
  void vitalsChanged(const broadcast::HpMpUpdate &hpMpUpdate);
  void characterLevelUpdate(int32_t level, int64_t expRequired);
  void characterExperienceUpdate(uint64_t currentExperience, uint32_t currentSpExperience);
  void characterSpUpdate(uint32_t skillPoints);
  void characterNameUpdate(const std::string &name);
  void inventoryGoldAmountUpdate(uint64_t goldAmount);
private:
  zmq::context_t &context_;
  std::atomic<bool> run_;
  std::thread thr_;
  void run();
  void handle(const broadcast::BroadcastMessage &message);
};

#endif // EVENT_HANDLER_HPP_