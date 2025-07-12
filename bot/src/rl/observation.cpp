#include "rl/observation.hpp"

#include <absl/strings/str_format.h>

#include <array>

namespace rl {

std::string Observation::toString() const {
  return absl::StrFormat("{hp:%d/%d, mp:%d/%d, opponentHp:%d/%d}", ourCurrentHp(),  ourMaxHp(), ourMpData_.currentValue(),  ourMpData_.maxValue(), opponentCurrentHp(),  opponentMaxHp());
}

size_t Observation::writeToArray(float *buffer) const {
  size_t bytesWritten = 0;
  for (const model_data::SkillModelData &skillModelData : skillData_) {
    bytesWritten += skillModelData.writeToArray(&buffer[bytesWritten]);
  }
  for (const model_data::ItemModelData &itemModelData : itemData_) {
    bytesWritten += itemModelData.writeToArray(&buffer[bytesWritten]);
  }
  bytesWritten += ourHpData_.writeToArray(&buffer[bytesWritten]);
  bytesWritten += ourMpData_.writeToArray(&buffer[bytesWritten]);
  bytesWritten += opponentHpData_.writeToArray(&buffer[bytesWritten]);
  return bytesWritten;
}

uint32_t Observation::ourCurrentHp() const {
  return ourHpData_.currentValue();
}

uint32_t Observation::ourMaxHp() const {
  return ourHpData_.maxValue();
}

uint32_t Observation::opponentCurrentHp() const {
  return opponentHpData_.currentValue();
}

uint32_t Observation::opponentMaxHp() const {
  return opponentHpData_.maxValue();
}

}