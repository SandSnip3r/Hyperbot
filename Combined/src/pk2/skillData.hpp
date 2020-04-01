#ifndef PK2_MEDIA_SKILL_DATA_HPP
#define PK2_MEDIA_SKILL_DATA_HPP

#include "../../../common/pk2/ref/skill.hpp"

#include <unordered_map>

namespace pk2 {

class SkillData {
public:
	using SkillMap = std::unordered_map<ref::SkillId,ref::Skill>;
	void addSkill(ref::Skill &&skill);
	bool haveSkillWithId(ref::SkillId id) const;
	const ref::Skill& getSkillById(ref::SkillId id) const;
	const SkillMap::size_type size() const;
private:
	SkillMap skills_;
};

} // namespace pk2

#endif // PK2_MEDIA_SKILL_DATA_HPP