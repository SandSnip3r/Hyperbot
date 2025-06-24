#include "event/event.hpp"
#include "rl/observation.hpp"

#include <absl/algorithm/container.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

#include <array>
#include <ostream>
#include <istream>

namespace rl {



std::string Observation::toString() const {
  return absl::StrFormat("{hp:%d/%d, mp:%d/%d, opponentHp:%d/%d, opponentMp:%d/%d, skillCooldowns:[%s], itemCooldowns:[%s]}", ourCurrentHp_,  ourMaxHp_, ourCurrentMp_,  ourMaxMp_, opponentCurrentHp_,  opponentMaxHp_, opponentCurrentMp_,  opponentMaxMp_, absl::StrJoin(skillCooldowns_, ", "), absl::StrJoin(itemCooldowns_, ", "));
}

template<typename T>
static void writeBinary(std::ostream &out, const T &value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

template<typename T>
static void readBinary(std::istream &in, T &value) {
  in.read(reinterpret_cast<char *>(&value), sizeof(T));
}

void Observation::saveToStream(std::ostream &out) const {
  using namespace std::chrono;
  const int64_t ts = duration_cast<nanoseconds>(timestamp_.time_since_epoch()).count();
  writeBinary(out, ts);
  writeBinary(out, static_cast<int>(eventCode_));
  writeBinary(out, ourCurrentHp_);
  writeBinary(out, ourMaxHp_);
  writeBinary(out, ourCurrentMp_);
  writeBinary(out, ourMaxMp_);
  writeBinary(out, weAreKnockedDown_);
  writeBinary(out, opponentCurrentHp_);
  writeBinary(out, opponentMaxHp_);
  writeBinary(out, opponentCurrentMp_);
  writeBinary(out, opponentMaxMp_);
  writeBinary(out, opponentIsKnockedDown_);
  writeBinary(out, hpPotionCount_);
  for (float v : remainingTimeOurBuffs_) writeBinary(out, v);
  for (float v : remainingTimeOpponentBuffs_) writeBinary(out, v);
  for (float v : remainingTimeOurDebuffs_) writeBinary(out, v);
  for (float v : remainingTimeOpponentDebuffs_) writeBinary(out, v);
  for (float v : skillCooldowns_) writeBinary(out, v);
  for (float v : itemCooldowns_) writeBinary(out, v);
}

Observation Observation::loadFromStream(std::istream &in) {
  Observation obs;
  using namespace std::chrono;
  int64_t ts;
  readBinary(in, ts);
  obs.timestamp_ = steady_clock::time_point(nanoseconds(ts));
  int ev;
  readBinary(in, ev);
  obs.eventCode_ = static_cast<event::EventCode>(ev);
  readBinary(in, obs.ourCurrentHp_);
  readBinary(in, obs.ourMaxHp_);
  readBinary(in, obs.ourCurrentMp_);
  readBinary(in, obs.ourMaxMp_);
  readBinary(in, obs.weAreKnockedDown_);
  readBinary(in, obs.opponentCurrentHp_);
  readBinary(in, obs.opponentMaxHp_);
  readBinary(in, obs.opponentCurrentMp_);
  readBinary(in, obs.opponentMaxMp_);
  readBinary(in, obs.opponentIsKnockedDown_);
  readBinary(in, obs.hpPotionCount_);
  for (float &v : obs.remainingTimeOurBuffs_) readBinary(in, v);
  for (float &v : obs.remainingTimeOpponentBuffs_) readBinary(in, v);
  for (float &v : obs.remainingTimeOurDebuffs_) readBinary(in, v);
  for (float &v : obs.remainingTimeOpponentDebuffs_) readBinary(in, v);
  for (float &v : obs.skillCooldowns_) readBinary(in, v);
  for (float &v : obs.itemCooldowns_) readBinary(in, v);
  return obs;
}

}