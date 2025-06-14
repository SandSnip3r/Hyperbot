#ifndef PK2_MEDIA_CHARACTER_DATA_HPP_
#define PK2_MEDIA_CHARACTER_DATA_HPP_

#include <silkroad_lib/pk2/ref/character.hpp>

#include <unordered_map>

namespace sro::pk2 {

class CharacterData {
public:
	using CharacterMap = std::unordered_map<sro::pk2::ref::CharacterId,sro::pk2::ref::Character>;
	void addCharacter(sro::pk2::ref::Character &&character);
	bool haveCharacterWithId(sro::pk2::ref::CharacterId id) const;
	const sro::pk2::ref::Character& getCharacterById(sro::pk2::ref::CharacterId id) const;
	const CharacterMap::size_type size() const;
private:
	CharacterMap characters_;
};

} // namespace sro::pk2

#endif // PK2_MEDIA_CHARACTER_DATA_HPP_