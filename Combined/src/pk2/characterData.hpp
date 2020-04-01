#ifndef PK2_MEDIA_CHARACTER_DATA_HPP_
#define PK2_MEDIA_CHARACTER_DATA_HPP_

#include "../../../common/pk2/ref/character.hpp"

#include <unordered_map>

namespace pk2 {

class CharacterData {
public:
	using CharacterMap = std::unordered_map<ref::CharacterId,ref::Character>;
	void addCharacter(ref::Character &&character);
	bool haveCharacterWithId(ref::CharacterId id) const;
	const ref::Character& getCharacterById(ref::CharacterId id) const;
	const CharacterMap::size_type size() const;
private:
	CharacterMap characters_;
};

} // namespace pk2

#endif // PK2_MEDIA_CHARACTER_DATA_HPP_