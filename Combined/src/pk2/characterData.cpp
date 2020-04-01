#include "characterData.hpp"

namespace pk2 {

void CharacterData::addCharacter(ref::Character &&character) {
  characters_.emplace(character.id, character);
}

bool CharacterData::haveCharacterWithId(ref::CharacterId id) const {
  return (characters_.find(id) != characters_.end());
}

const ref::Character& CharacterData::getCharacterById(ref::CharacterId id) const {
  auto it = characters_.find(id);
  if (it == characters_.end()) {
    throw std::runtime_error("Trying to get non-existent character with id "+std::to_string(id));
  }
  return it->second;
}

const CharacterData::CharacterMap::size_type CharacterData::size() const {
  return characters_.size();
}

} // namespace pk2