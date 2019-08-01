#ifndef PK2_MEDIA_SKILL_DATA_HPP
#define PK2_MEDIA_SKILL_DATA_HPP

#include "../../common/skill.hpp"

#include <unordered_map>

namespace pk2::media {

class SkillData {
public:
	using SkillMap = std::unordered_map<SkillId,Skill>;
	void addSkill(Skill &&skill);
	bool haveSkillWithId(SkillId id) const;
	const Skill& getSkillById(SkillId id) const;
	const SkillMap::size_type size() const;
private:
	SkillMap skills_;
};

} // namespace pk2::media

#endif // PK2_MEDIA_SKILL_DATA_HPP