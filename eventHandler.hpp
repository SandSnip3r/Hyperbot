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
  void message3Received(const double val);
  void vitalsChanged(const broadcast::HpMpUpdate &hpMpUpdate);
private:
  zmq::context_t &context_;
  std::atomic<bool> run_;
  std::thread thr_;
  void run();
  void handle(const broadcast::BroadcastMessage &message);
};

#endif // EVENT_HANDLER_HPP_