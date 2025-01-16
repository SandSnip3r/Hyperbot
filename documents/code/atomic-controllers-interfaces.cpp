class ChatMessage {};
class UserId {};
class MobId {};
class PetId {};
class Notification {};
class Location {};
class Item {};
class PartyInvite {};
class PartySettings {};
class PartyMatching {};
class MatchingJoinRequest {};
using GoldInt = int64_t;
using HpMpInt = uint32_t;
using SpInt = uint32_t;
using ExpDouble = double;

// =================Chat=================
// Events:
void messageReceived(ChatMessage msg);
// Interface:
void sendMessage(ChatMessage msg);
void sendMessageAsync(ChatMessage msg);
void blockUser(UserId id);
void unblockUser(UserId id);

// =============Notification=============
/* Useful for notices from a GM, notices from the gameserver, unique monster spawns, bot messages to the player */
// Events:
void notificationReceived(Notification notification);
// Interface:
void injectClientNotification(Notification notification);

// ==============StatPoints==============
// Events:
void remainingStatPoints(int n);
void gainedStatPoints(int n);
// Interface:
bool allocateInt(int n);
bool allocateStr(int n);

// ===============Berzerk================
// Events:
// Interface:
bool activateBerzerk();
double berzerkGauge();

// ===============Movement===============
// Events:
void monsterMoved(MobId id, Location location);
void petMoved(PetId id, Location location);
void playerMoved(UserId id, Location location);
// Interface:
bool move(Location location);
void moveAsync(Location location);

// ================Vitals================
// Events:
void monsterHealthChanged(MobId id, HpMpInt value);
void petHealthChanged(PetId id, HpMpInt value);
void playerHealthChanged(UserId id, HpMpInt value);
// Interface:

// ==============Experience==============
// Events:
void playerExperienceChanged(ExpDouble value);
void playerSPChange(SpInt spCount);
void petExperienceChanged(ExpDouble value);
// Interface:

// ================Stall=================
// Events:
void playerEnteredStall(UserId id);
// Interface:
void postItem(Item item, GoldInt price);
void deleteItem(Item item);
void updateItem(Item item, GoldInt price);
void openStall();
void modifyStall();
void exitStall();

// ================Party=================
// Events:
void inviteReceived(PartyInvite invite);
void playerJoined(UserId id);
void matchingJoinRequest(MatchingJoinRequest request);
// Interface:
void setPartySettings(PartySettings settings);
void forfeitLeadership(UserId id);
void leave();
void invitePlayer(UserId id);
void matchingPost(PartyMatching details);
void matchingDelete();


// items dropped on the ground
// mobs' spawn, location, and health
// Item movement in inventory
// Skill updates
// Alchemy
// Pet
// Interface with NPC
// Equiping items
// Using item
// Attack
// Silk
// Teleport
// Repair
// Quit game
// Academy
// Quests