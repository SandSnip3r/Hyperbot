#ifndef PK2_MEDIA_CHARACTER_DATA_HPP_
#define PK2_MEDIA_CHARACTER_DATA_HPP_

#include "../../common/characterInfo.hpp"

#include <unordered_map>

namespace pk2::media {

class CharacterData {
public:
	using CharacterMap = std::unordered_map<CharacterId,Character>;
	void addCharacter(Character &&character);
	bool haveCharacterWithId(CharacterId id) const;
	const Character& getCharacterById(CharacterId id) const;
	const CharacterMap::size_type size() const;
private:
	CharacterMap characters_;
};

} // namespace pk2::media

#endif // PK2_MEDIA_CHARACTER_DATA_HPP_