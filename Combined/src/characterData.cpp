#include "characterData.hpp"

namespace pk2::media {

void CharacterData::addCharacter(Character &&character) {
  characters_.emplace(character.id, character);
}

bool CharacterData::haveCharacterWithId(CharacterId id) const {
  return (characters_.find(id) != characters_.end());
}

const Character& CharacterData::getCharacterById(CharacterId id) const {
  auto it = characters_.find(id);
  if (it == characters_.end()) {
    throw std::runtime_error("Trying to get non-existent character with id "+std::to_string(id));
  }
  return it->second;
}

const CharacterData::CharacterMap::size_type CharacterData::size() const {
  return characters_.size();
}

} // namespace pk2::media