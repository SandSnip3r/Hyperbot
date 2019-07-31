#include "opcode.hpp"

namespace OpcodeHelp {
std::string toStr(Opcode opcode) {
  if (opcode == Opcode::kServerEnvironmentWeather) {
    return "kServerEnvironmentWeather";
  }
  if (opcode == Opcode::HACK) {
    return "HACK";
  }
  if (opcode == Opcode::LOGIN_CLIENT_INFO) {
    return "LOGIN_CLIENT_INFO";
  }
  if (opcode == Opcode::LOGIN_CLIENT_KEEP_ALIVE) {
    return "LOGIN_CLIENT_KEEP_ALIVE";
  }
  if (opcode == Opcode::LOGIN_CLIENT_PATCH_REQUEST) {
    return "LOGIN_CLIENT_PATCH_REQUEST";
  }
  if (opcode == Opcode::LOGIN_CLIENT_SERVERLIST_REQUEST) {
    return "LOGIN_CLIENT_SERVERLIST_REQUEST";
  }
  if (opcode == Opcode::LOGIN_CLIENT_AUTH) {
    return "LOGIN_CLIENT_AUTH";
  }
  if (opcode == Opcode::LOGIN_CLIENT_ACCEPT_HANDSHAKE) {
    return "LOGIN_CLIENT_ACCEPT_HANDSHAKE";
  }
  if (opcode == Opcode::LOGIN_CLIENT_LAUNCHER) {
    return "LOGIN_CLIENT_LAUNCHER";
  }
  if (opcode == Opcode::LOGIN_SERVER_INFO) {
    return "LOGIN_SERVER_INFO";
  }
  if (opcode == Opcode::LOGIN_SERVER_HANDSHAKE) {
    return "LOGIN_SERVER_HANDSHAKE";
  }
  if (opcode == Opcode::LOGIN_SERVER_PATCH_INFO) {
    return "LOGIN_SERVER_PATCH_INFO";
  }
  if (opcode == Opcode::LOGIN_SERVER_LAUNCHER) {
    return "LOGIN_SERVER_LAUNCHER";
  }
  if (opcode == Opcode::LOGIN_SERVER_LIST) {
    return "LOGIN_SERVER_LIST";
  }
  if (opcode == Opcode::LOGIN_SERVER_AUTH_INFO) {
    return "LOGIN_SERVER_AUTH_INFO";
  }
  if (opcode == Opcode::CLIENT_INFO) {
    return "CLIENT_INFO";
  }
  if (opcode == Opcode::CLIENT_ACCEPT_HANDSHAKE) {
    return "CLIENT_ACCEPT_HANDSHAKE";
  }
  if (opcode == Opcode::CLIENT_KEEP_ALIVE) {
    return "CLIENT_KEEP_ALIVE";
  }
  if (opcode == Opcode::CLIENT_PATCH_REQUEST) {
    return "CLIENT_PATCH_REQUEST";
  }
  if (opcode == Opcode::CLIENT_AUTH) {
    return "CLIENT_AUTH";
  }
  if (opcode == Opcode::CLIENT_ITEM_MOVE) {
    return "CLIENT_ITEM_MOVE";
  }
  if (opcode == Opcode::CLIENT_INGAME_NOTIFY) {
    return "CLIENT_INGAME_NOTIFY";
  }
  if (opcode == Opcode::CLIENT_CLOSE) {
    return "CLIENT_CLOSE";
  }
  if (opcode == Opcode::CLIENT_COUNTDOWN_INTERRUPT) {
    return "CLIENT_COUNTDOWN_INTERRUPT";
  }
  if (opcode == Opcode::CLIENT_CHARACTER) {
    return "CLIENT_CHARACTER";
  }
  if (opcode == Opcode::CLIENT_CHAT) {
    return "CLIENT_CHAT";
  }
  if (opcode == Opcode::CLIENT_INGAME_REQUEST) {
    return "CLIENT_INGAME_REQUEST";
  }
  if (opcode == Opcode::CLIENT_TARGET) {
    return "CLIENT_TARGET";
  }
  if (opcode == Opcode::CLIENT_GM) {
    return "CLIENT_GM";
  }
  if (opcode == Opcode::CLIENT_MOVEMENT) {
    return "CLIENT_MOVEMENT";
  }
  if (opcode == Opcode::CLIENT_TRANSPORT_MOVE) {
    return "CLIENT_TRANSPORT_MOVE";
  }
  if (opcode == Opcode::CLIENT_PLAYER_ACTION) {
    return "CLIENT_PLAYER_ACTION";
  }
  if (opcode == Opcode::CLIENT_STR_UPDATE) {
    return "CLIENT_STR_UPDATE";
  }
  if (opcode == Opcode::CLIENT_INT_UPDATE) {
    return "CLIENT_INT_UPDATE";
  }
  if (opcode == Opcode::CLIENT_CHARACTER_STATE) {
    return "CLIENT_CHARACTER_STATE";
  }
  if (opcode == Opcode::CLIENT_RESPAWN) {
    return "CLIENT_RESPAWN";
  }
  if (opcode == Opcode::CLIENT_MASTERYUPDATE) {
    return "CLIENT_MASTERYUPDATE";
  }
  if (opcode == Opcode::CLIENT_SKILLUPDATE) {
    return "CLIENT_SKILLUPDATE";
  }
  if (opcode == Opcode::CLIENT_EMOTION) {
    return "CLIENT_EMOTION";
  }
  if (opcode == Opcode::CLIENT_ITEM_USE) {
    return "CLIENT_ITEM_USE";
  }
  if (opcode == Opcode::CLIENT_HOTKEY_CHANGE) {
    return "CLIENT_HOTKEY_CHANGE";
  }
  if (opcode == Opcode::CLIENT_OPEN_SHOP) {
    return "CLIENT_OPEN_SHOP";
  }
  if (opcode == Opcode::CLIENT_CLOSE_SHOP) {
    return "CLIENT_CLOSE_SHOP";
  }
  if (opcode == Opcode::CLIENT_TELEPORT) {
    return "CLIENT_TELEPORT";
  }
  if (opcode == Opcode::CLIENT_PARTY_FORM) {
    return "CLIENT_PARTY_FORM";
  }
  if (opcode == Opcode::CLIENT_PARTY_EDIT) {
    return "CLIENT_PARTY_EDIT";
  }
  if (opcode == Opcode::CLIENT_PARTY_DELETE) {
    return "CLIENT_PARTY_DELETE";
  }
  if (opcode == Opcode::CLIENT_PARTY_MATCHING) {
    return "CLIENT_PARTY_MATCHING";
  }
  if (opcode == Opcode::CLIENT_PARTY_REQUEST) {
    return "CLIENT_PARTY_REQUEST";
  }
  if (opcode == Opcode::CLIENT_PARTY_ACCEPT) {
    return "CLIENT_PARTY_ACCEPT";
  }
  if (opcode == Opcode::CLIENT_PARTY_INVITE) {
    return "CLIENT_PARTY_INVITE";
  }
  if (opcode == Opcode::CLIENT_PARTY_DISMISS) {
    return "CLIENT_PARTY_DISMISS";
  }
  if (opcode == Opcode::CLIENT_PARTY_KICK) {
    return "CLIENT_PARTY_KICK";
  }
  if (opcode == Opcode::CLIENT_ANIMATION_INVITE) {
    return "CLIENT_ANIMATION_INVITE";
  }
  if (opcode == Opcode::CLIENT_ALCHEMY) {
    return "CLIENT_ALCHEMY";
  }
  if (opcode == Opcode::CLIENT_ALCHEMYSTONE) {
    return "CLIENT_ALCHEMYSTONE";
  }
  if (opcode == Opcode::CLIENT_TRANSPORT_HOME) {
    return "CLIENT_TRANSPORT_HOME";
  }
  if (opcode == Opcode::CLIENT_TRANSPORT_DELETE) {
    return "CLIENT_TRANSPORT_DELETE";
  }
  if (opcode == Opcode::CLIENT_OPEN_STORAGE) {
    return "CLIENT_OPEN_STORAGE";
  }
  if (opcode == Opcode::CLIENT_REPAIR) {
    return "CLIENT_REPAIR";
  }
  if (opcode == Opcode::CLIENT_USE_BERSERK) {
    return "CLIENT_USE_BERSERK";
  }
  if (opcode == Opcode::SERVER_INFO) {
    return "SERVER_INFO";
  }
  if (opcode == Opcode::SERVER_HANDSHAKE) {
    return "SERVER_HANDSHAKE";
  }
  if (opcode == Opcode::SERVER_PATCH_INFO) {
    return "SERVER_PATCH_INFO";
  }
  if (opcode == Opcode::SERVER_LOGIN_RESULT) {
    return "SERVER_LOGIN_RESULT";
  }
  if (opcode == Opcode::SERVER_CHARACTER) {
    return "SERVER_CHARACTER";
  }
  if (opcode == Opcode::SERVER_CHARDATA) {
    return "SERVER_CHARDATA";
  }
  if (opcode == Opcode::SERVER_INGAME_ACCEPT) {
    return "SERVER_INGAME_ACCEPT";
  }
  if (opcode == Opcode::SERVER_LOADING_START) {
    return "SERVER_LOADING_START";
  }
  if (opcode == Opcode::SERVER_LOADING_END) {
    return "SERVER_LOADING_END";
  }
  if (opcode == Opcode::SERVER_WORLD_CLOCK) {
    return "SERVER_WORLD_CLOCK";
  }
  if (opcode == Opcode::SERVER_SPAWN) {
    return "SERVER_SPAWN";
  }
  if (opcode == Opcode::SERVER_DESPAWN) {
    return "SERVER_DESPAWN";
  }
  if (opcode == Opcode::SERVER_GROUPSPAWN_HEAD) {
    return "SERVER_GROUPSPAWN_HEAD";
  }
  if (opcode == Opcode::SERVER_GROUPSPAWN_BODY) {
    return "SERVER_GROUPSPAWN_BODY";
  }
  if (opcode == Opcode::SERVER_GROUPSPAWN_TAIL) {
    return "SERVER_GROUPSPAWN_TAIL";
  }
  if (opcode == Opcode::SERVER_ITEM_EQUIP) {
    return "SERVER_ITEM_EQUIP";
  }
  if (opcode == Opcode::SERVER_ITEM_UNEQUIP) {
    return "SERVER_ITEM_UNEQUIP";
  }
  if (opcode == Opcode::SERVER_ITEM_MOVEMENT) {
    return "SERVER_ITEM_MOVEMENT";
  }
  if (opcode == Opcode::SERVER_NEW_GOLD_AMOUNT) {
    return "SERVER_NEW_GOLD_AMOUNT";
  }
  if (opcode == Opcode::SERVER_ANIMATION_ITEM_PICKUP) {
    return "SERVER_ANIMATION_ITEM_PICKUP";
  }
  if (opcode == Opcode::SERVER_ITEM_USE) {
    return "SERVER_ITEM_USE";
  }
  if (opcode == Opcode::SERVER_ANIMATION_ITEM_USE) {
    return "SERVER_ANIMATION_ITEM_USE";
  }
  if (opcode == Opcode::SERVER_ANIMATION_CAPE) {
    return "SERVER_ANIMATION_CAPE";
  }
  if (opcode == Opcode::SERVER_ITEM_QUANTITY_UPDATE) {
    return "SERVER_ITEM_QUANTITY_UPDATE";
  }
  if (opcode == Opcode::SERVER_QUIT_GAME) {
    return "SERVER_QUIT_GAME";
  }
  if (opcode == Opcode::SERVER_COUNTDOWN) {
    return "SERVER_COUNTDOWN";
  }
  if (opcode == Opcode::SERVER_COUNTDOWN_INTERRUPT) {
    return "SERVER_COUNTDOWN_INTERRUPT";
  }
  if (opcode == Opcode::SERVER_STATS) {
    return "SERVER_STATS";
  }
  if (opcode == Opcode::SERVER_STR_UPDATE) {
    return "SERVER_STR_UPDATE";
  }
  if (opcode == Opcode::SERVER_INT_UPDATE) {
    return "SERVER_INT_UPDATE";
  }
  if (opcode == Opcode::SERVER_CHARACTER_STATE) {
    return "SERVER_CHARACTER_STATE";
  }
  if (opcode == Opcode::SERVER_HPMP_UPDATE) {
    return "SERVER_HPMP_UPDATE";
  }
  if (opcode == Opcode::SERVER_ANIMATION_LEVEL_UP) {
    return "SERVER_ANIMATION_LEVEL_UP";
  }
  if (opcode == Opcode::SERVER_EXP) {
    return "SERVER_EXP";
  }
  if (opcode == Opcode::SERVER_MASTERYUPDATE) {
    return "SERVER_MASTERYUPDATE";
  }
  if (opcode == Opcode::SERVER_SKILLPOINTS) {
    return "SERVER_SKILLPOINTS";
  }
  if (opcode == Opcode::SERVER_SKILLUPDATE) {
    return "SERVER_SKILLUPDATE";
  }
  if (opcode == Opcode::SERVER_CHAT) {
    return "SERVER_CHAT";
  }
  if (opcode == Opcode::SERVER_CHAT_ACCEPT) {
    return "SERVER_CHAT_ACCEPT";
  }
  if (opcode == Opcode::SERVER_TARGET) {
    return "SERVER_TARGET";
  }
  if (opcode == Opcode::SERVER_MOVEMENT) {
    return "SERVER_MOVEMENT";
  }
  if (opcode == Opcode::SERVER_UNIQUE) {
    return "SERVER_UNIQUE";
  }
  if (opcode == Opcode::SERVER_ANIMATION_COS_SPAWN) {
    return "SERVER_ANIMATION_COS_SPAWN";
  }
  if (opcode == Opcode::SERVER_COS_SIT_UP) {
    return "SERVER_COS_SIT_UP";
  }
  if (opcode == Opcode::SERVER_ANIMATION_COS_REMOVE_MENU) {
    return "SERVER_ANIMATION_COS_REMOVE_MENU";
  }
  if (opcode == Opcode::SERVER_COS_DELETE) {
    return "SERVER_COS_DELETE";
  }
  if (opcode == Opcode::SERVER_ATTACK) {
    return "SERVER_ATTACK";
  }
  if (opcode == Opcode::SERVER_SKILL_ATTACK) {
    return "SERVER_SKILL_ATTACK";
  }
  if (opcode == Opcode::SERVER_END_SKILL) {
    return "SERVER_END_SKILL";
  }
  if (opcode == Opcode::SERVER_BUFF_START) {
    return "SERVER_BUFF_START";
  }
  if (opcode == Opcode::SERVER_BUFF_END) {
    return "SERVER_BUFF_END";
  }
  if (opcode == Opcode::SERVER_DEAD) {
    return "SERVER_DEAD";
  }
  if (opcode == Opcode::SERVER_DEAD2) {
    return "SERVER_DEAD2";
  }
  if (opcode == Opcode::SERVER_PARTY_FORM) {
    return "SERVER_PARTY_FORM";
  }
  if (opcode == Opcode::SERVER_PARTY_EDIT) {
    return "SERVER_PARTY_EDIT";
  }
  if (opcode == Opcode::SERVER_PARTY_DELETE) {
    return "SERVER_PARTY_DELETE";
  }
  if (opcode == Opcode::SERVER_PARTY_MATCHING) {
    return "SERVER_PARTY_MATCHING";
  }
  if (opcode == Opcode::SERVER_PARTY_ACCEPT) {
    return "SERVER_PARTY_ACCEPT";
  }
  if (opcode == Opcode::SERVER_PARTY_REQUEST) {
    return "SERVER_PARTY_REQUEST";
  }
  if (opcode == Opcode::SERVER_PARTY_NEW_PARTY) {
    return "SERVER_PARTY_NEW_PARTY";
  }
  if (opcode == Opcode::SERVER_PARTY_CHANGES) {
    return "SERVER_PARTY_CHANGES";
  }
  if (opcode == Opcode::SERVER_PARTY_INVITE) {
    return "SERVER_PARTY_INVITE";
  }
  if (opcode == Opcode::SERVER_ANIMATION_INVITE) {
    return "SERVER_ANIMATION_INVITE";
  }
  if (opcode == Opcode::SERVER_OPEN_SHOP) {
    return "SERVER_OPEN_SHOP";
  }
  if (opcode == Opcode::SERVER_CLOSE_SHOP) {
    return "SERVER_CLOSE_SHOP";
  }
  if (opcode == Opcode::SERVER_SILK_AMOUNT) {
    return "SERVER_SILK_AMOUNT";
  }
  if (opcode == Opcode::SERVER_TELEPORT) {
    return "SERVER_TELEPORT";
  }
  if (opcode == Opcode::SERVER_ANIMATION_TELEPORT) {
    return "SERVER_ANIMATION_TELEPORT";
  }
  if (opcode == Opcode::SERVER_STORAGE_GOLD) {
    return "SERVER_STORAGE_GOLD";
  }
  if (opcode == Opcode::SERVER_STORAGE_ITEMS) {
    return "SERVER_STORAGE_ITEMS";
  }
  if (opcode == Opcode::SERVER_STORAGE_END) {
    return "SERVER_STORAGE_END";
  }
  if (opcode == Opcode::SERVER_ALCHEMY) {
    return "SERVER_ALCHEMY";
  }
  if (opcode == Opcode::SERVER_ALCHEMYSTONE) {
    return "SERVER_ALCHEMYSTONE";
  }
  if (opcode == Opcode::SERVER_REPAIR) {
    return "SERVER_REPAIR";
  }
  if (opcode == Opcode::SERVER_ITEM_DURABILITY_CHANGE) {
    return "SERVER_ITEM_DURABILITY_CHANGE";
  }
  if (opcode == Opcode::SERVER_CHARACTER_STUCK) {
    return "SERVER_CHARACTER_STUCK";
  }
  return "UNKNOWN";
}
} // namespace OpcodeHelp