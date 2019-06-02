#include "bot.hpp"
#include "opcode.hpp"

#include <iostream>

Bot::Bot(std::function<void(PacketContainer&, PacketContainer::Direction)> injectionFunction) : injectionFunction_(injectionFunction), loginModule_(injectionFunction_) {
}

/* void Bot::configure(Config &config) {
  //Maybe pass configuration down to each module?
} */

bool Bot::packetReceived(const PacketContainer &packet, PacketContainer::Direction packetDirection) {
  //Proxy called this because a new packet arrived
  bool forwardPacket=true;
  Opcode packetOpcode = static_cast<Opcode>(packet.opcode);
  switch (packetOpcode) {
    case Opcode::CLIENT_CAFE:
      std::cout << "CLIENT_CAFE\n";
      loginModule_.cafeSent();
      break;
    case Opcode::LOGIN_CLIENT_INFO:
    // case Opcode::LOGIN_SERVER_INFO:
    // case Opcode::CLIENT_INFO:
    // case Opcode::SERVER_INFO:
      std::cout << "LOGIN_CLIENT_INFO, LOGIN_SERVER_INFO, CLIENT_INFO, SERVER_INFO\n";
      forwardPacket = loginModule_.loginClientInfo(packet);
      break;
    case Opcode::LOGIN_CLIENT_KEEP_ALIVE:
    // case Opcode::CLIENT_KEEP_ALIVE:
      std::cout << "LOGIN_CLIENT_KEEP_ALIVE, CLIENT_KEEP_ALIVE\n";
      break;
    case Opcode::LOGIN_CLIENT_PATCH_REQUEST:
    // case Opcode::CLIENT_PATCH_REQUEST:
      std::cout << "LOGIN_CLIENT_PATCH_REQUEST, CLIENT_PATCH_REQUEST\n";
      break;
    case Opcode::LOGIN_CLIENT_SERVERLIST_REQUEST:
      std::cout << "LOGIN_CLIENT_SERVERLIST_REQUEST\n";
      break;
    case Opcode::LOGIN_CLIENT_AUTH:
      std::cout << "LOGIN_CLIENT_AUTH\n";
      break;
    case Opcode::LOGIN_CLIENT_ACCEPT_HANDSHAKE:
    // case Opcode::CLIENT_ACCEPT_HANDSHAKE:
      std::cout << "LOGIN_CLIENT_ACCEPT_HANDSHAKE, CLIENT_ACCEPT_HANDSHAKE\n";
      break;
    case Opcode::LOGIN_CLIENT_LAUNCHER:
      std::cout << "LOGIN_CLIENT_LAUNCHER\n";
      break;
    case Opcode::LOGIN_SERVER_HANDSHAKE:
    // case Opcode::SERVER_HANDSHAKE:
      std::cout << "LOGIN_SERVER_HANDSHAKE, SERVER_HANDSHAKE\n";
      break;
    case Opcode::LOGIN_SERVER_PATCH_INFO:
    // case Opcode::LOGIN_SERVER_LAUNCHER:
    // case Opcode::SERVER_PATCH_INFO:
      std::cout << "LOGIN_SERVER_PATCH_INFO, LOGIN_SERVER_LAUNCHER, SERVER_PATCH_INFO\n";
      break;
    case Opcode::LOGIN_SERVER_LIST:
      std::cout << "LOGIN_SERVER_LIST\n";
      loginModule_.serverListReceived(packet);
      break;
    case Opcode::LOGIN_SERVER_AUTH_INFO:
      std::cout << "LOGIN_SERVER_AUTH_INFO\n";
      loginModule_.serverAuthInfoReceived(packet);
      break;
    case Opcode::CLIENT_AUTH:
      std::cout << "CLIENT_AUTH\n";
      break;
    case Opcode::CLIENT_ITEM_MOVE:
      std::cout << "CLIENT_ITEM_MOVE\n";
      break;
    case Opcode::CLIENT_INGAME_NOTIFY:
      std::cout << "CLIENT_INGAME_NOTIFY\n";
      break;
    case Opcode::CLIENT_CLOSE:
      std::cout << "CLIENT_CLOSE\n";
      break;
    case Opcode::CLIENT_COUNTDOWN_INTERRUPT:
      std::cout << "CLIENT_COUNTDOWN_INTERRUPT\n";
      break;
    case Opcode::CLIENT_CHARACTER:
      std::cout << "CLIENT_CHARACTER\n";
      break;
    case Opcode::CLIENT_CHAT:
      std::cout << "CLIENT_CHAT\n";
      break;
    case Opcode::CLIENT_INGAME_REQUEST:
      std::cout << "CLIENT_INGAME_REQUEST\n";
      break;
    case Opcode::CLIENT_TARGET:
      std::cout << "CLIENT_TARGET\n";
      break;
    case Opcode::CLIENT_GM:
      std::cout << "CLIENT_GM\n";
      break;
    case Opcode::CLIENT_MOVEMENT:
      std::cout << "CLIENT_MOVEMENT\n";
      break;
    case Opcode::CLIENT_TRANSPORT_MOVE:
      std::cout << "CLIENT_TRANSPORT_MOVE\n";
      break;
    case Opcode::CLIENT_PLAYER_ACTION:
      std::cout << "CLIENT_PLAYER_ACTION\n";
      break;
    case Opcode::CLIENT_STR_UPDATE:
      std::cout << "CLIENT_STR_UPDATE\n";
      break;
    case Opcode::CLIENT_INT_UPDATE:
      std::cout << "CLIENT_INT_UPDATE\n";
      break;
    case Opcode::CLIENT_CHARACTER_STATE:
      std::cout << "CLIENT_CHARACTER_STATE\n";
      break;
    case Opcode::CLIENT_RESPAWN:
      std::cout << "CLIENT_RESPAWN\n";
      break;
    case Opcode::CLIENT_MASTERYUPDATE:
      std::cout << "CLIENT_MASTERYUPDATE\n";
      break;
    case Opcode::CLIENT_SKILLUPDATE:
      std::cout << "CLIENT_SKILLUPDATE\n";
      break;
    case Opcode::CLIENT_EMOTION:
      std::cout << "CLIENT_EMOTION\n";
      break;
    case Opcode::CLIENT_ITEM_USE:
      std::cout << "CLIENT_ITEM_USE\n";
      break;
    case Opcode::CLIENT_HOTKEY_CHANGE:
      std::cout << "CLIENT_HOTKEY_CHANGE\n";
      break;
    case Opcode::CLIENT_OPEN_SHOP:
      std::cout << "CLIENT_OPEN_SHOP\n";
      break;
    case Opcode::CLIENT_CLOSE_SHOP:
      std::cout << "CLIENT_CLOSE_SHOP\n";
      break;
    case Opcode::CLIENT_TELEPORT:
      std::cout << "CLIENT_TELEPORT\n";
      break;
    case Opcode::CLIENT_PARTY_FORM:
      std::cout << "CLIENT_PARTY_FORM\n";
      break;
    case Opcode::CLIENT_PARTY_EDIT:
      std::cout << "CLIENT_PARTY_EDIT\n";
      break;
    case Opcode::CLIENT_PARTY_DELETE:
      std::cout << "CLIENT_PARTY_DELETE\n";
      break;
    case Opcode::CLIENT_PARTY_MATCHING:
      std::cout << "CLIENT_PARTY_MATCHING\n";
      break;
    case Opcode::CLIENT_PARTY_REQUEST:
    // case Opcode::SERVER_PARTY_REQUEST:
      std::cout << "CLIENT_PARTY_REQUEST, SERVER_PARTY_REQUEST\n";
      break;
    case Opcode::CLIENT_PARTY_ACCEPT:
      std::cout << "CLIENT_PARTY_ACCEPT\n";
      break;
    case Opcode::CLIENT_PARTY_INVITE:
      std::cout << "CLIENT_PARTY_INVITE\n";
      break;
    case Opcode::CLIENT_PARTY_DISMISS:
      std::cout << "CLIENT_PARTY_DISMISS\n";
      break;
    case Opcode::CLIENT_PARTY_KICK:
      std::cout << "CLIENT_PARTY_KICK\n";
      break;
    case Opcode::CLIENT_ANIMATION_INVITE:
    // case Opcode::SERVER_ANIMATION_INVITE:
      std::cout << "CLIENT_ANIMATION_INVITE, SERVER_ANIMATION_INVITE\n";
      break;
    case Opcode::CLIENT_ALCHEMY:
      std::cout << "CLIENT_ALCHEMY\n";
      break;
    case Opcode::CLIENT_ALCHEMYSTONE:
      std::cout << "CLIENT_ALCHEMYSTONE\n";
      break;
    case Opcode::CLIENT_TRANSPORT_HOME:
    // case Opcode::CLIENT_TRANSPORT_DELETE:
      std::cout << "CLIENT_TRANSPORT_HOME, CLIENT_TRANSPORT_DELETE\n";
      break;
    case Opcode::CLIENT_OPEN_STORAGE:
      std::cout << "CLIENT_OPEN_STORAGE\n";
      break;
    case Opcode::CLIENT_REPAIR:
      std::cout << "CLIENT_REPAIR\n";
      break;
    case Opcode::CLIENT_USE_BERSERK:
      std::cout << "CLIENT_USE_BERSERK\n";
      break;
    case Opcode::SERVER_LOGIN_RESULT:
      std::cout << "SERVER_LOGIN_RESULT\n";
      loginModule_.serverLoginResultReceived(packet);
      break;
    case Opcode::SERVER_CHARACTER:
      std::cout << "SERVER_CHARACTER\n";
      loginModule_.serverCharacterListReceived(packet);
      break;
    case Opcode::SERVER_CHARDATA:
      std::cout << "SERVER_CHARDATA\n";
      break;
    case Opcode::SERVER_INGAME_ACCEPT:
      std::cout << "SERVER_INGAME_ACCEPT\n";
      break;
    case Opcode::SERVER_LOADING_START:
      std::cout << "SERVER_LOADING_START\n";
      break;
    case Opcode::SERVER_LOADING_END:
      std::cout << "SERVER_LOADING_END\n";
      break;
    case Opcode::SERVER_WORLD_CLOCK:
      std::cout << "SERVER_WORLD_CLOCK\n";
      break;
    case Opcode::SERVER_SPAWN:
      std::cout << "SERVER_SPAWN\n";
      break;
    case Opcode::SERVER_DESPAWN:
      std::cout << "SERVER_DESPAWN\n";
      break;
    case Opcode::SERVER_GROUPSPAWN_HEAD:
      std::cout << "SERVER_GROUPSPAWN_HEAD\n";
      break;
    case Opcode::SERVER_GROUPSPAWN_BODY:
      std::cout << "SERVER_GROUPSPAWN_BODY\n";
      break;
    case Opcode::SERVER_GROUPSPAWN_TAIL:
      std::cout << "SERVER_GROUPSPAWN_TAIL\n";
      break;
    case Opcode::SERVER_ITEM_EQUIP:
      std::cout << "SERVER_ITEM_EQUIP\n";
      break;
    case Opcode::SERVER_ITEM_UNEQUIP:
      std::cout << "SERVER_ITEM_UNEQUIP\n";
      break;
    case Opcode::SERVER_ITEM_MOVEMENT:
      std::cout << "SERVER_ITEM_MOVEMENT\n";
      break;
    case Opcode::SERVER_NEW_GOLD_AMOUNT:
    // case Opcode::SERVER_SKILLPOINTS:
      std::cout << "SERVER_NEW_GOLD_AMOUNT, SERVER_SKILLPOINTS\n";
      break;
    case Opcode::SERVER_ANIMATION_ITEM_PICKUP:
      std::cout << "SERVER_ANIMATION_ITEM_PICKUP\n";
      break;
    case Opcode::SERVER_ITEM_USE:
      std::cout << "SERVER_ITEM_USE\n";
      break;
    case Opcode::SERVER_ANIMATION_ITEM_USE:
      std::cout << "SERVER_ANIMATION_ITEM_USE\n";
      break;
    case Opcode::SERVER_ANIMATION_CAPE:
      std::cout << "SERVER_ANIMATION_CAPE\n";
      break;
    case Opcode::SERVER_ITEM_QUANTITY_UPDATE:
      std::cout << "SERVER_ITEM_QUANTITY_UPDATE\n";
      break;
    case Opcode::SERVER_QUIT_GAME:
      std::cout << "SERVER_QUIT_GAME\n";
      break;
    case Opcode::SERVER_COUNTDOWN:
      std::cout << "SERVER_COUNTDOWN\n";
      break;
    case Opcode::SERVER_COUNTDOWN_INTERRUPT:
      std::cout << "SERVER_COUNTDOWN_INTERRUPT\n";
      break;
    case Opcode::SERVER_STATS:
      std::cout << "SERVER_STATS\n";
      break;
    case Opcode::SERVER_STR_UPDATE:
      std::cout << "SERVER_STR_UPDATE\n";
      break;
    case Opcode::SERVER_INT_UPDATE:
      std::cout << "SERVER_INT_UPDATE\n";
      break;
    case Opcode::SERVER_CHARACTER_STATE:
      std::cout << "SERVER_CHARACTER_STATE\n";
      break;
    case Opcode::SERVER_HPMP_UPDATE:
      std::cout << "SERVER_HPMP_UPDATE\n";
      break;
    case Opcode::SERVER_ANIMATION_LEVEL_UP:
      std::cout << "SERVER_ANIMATION_LEVEL_UP\n";
      break;
    case Opcode::SERVER_EXP:
      std::cout << "SERVER_EXP\n";
      break;
    case Opcode::SERVER_MASTERYUPDATE:
      std::cout << "SERVER_MASTERYUPDATE\n";
      break;
    case Opcode::SERVER_SKILLUPDATE:
      std::cout << "SERVER_SKILLUPDATE\n";
      break;
    case Opcode::SERVER_CHAT:
      std::cout << "SERVER_CHAT\n";
      break;
    case Opcode::SERVER_CHAT_ACCEPT:
      std::cout << "SERVER_CHAT_ACCEPT\n";
      break;
    case Opcode::SERVER_TARGET:
      std::cout << "SERVER_TARGET\n";
      break;
    case Opcode::SERVER_MOVEMENT:
      std::cout << "SERVER_MOVEMENT\n";
      break;
    case Opcode::SERVER_UNIQUE:
      std::cout << "SERVER_UNIQUE\n";
      break;
    case Opcode::SERVER_ANIMATION_COS_SPAWN:
      std::cout << "SERVER_ANIMATION_COS_SPAWN\n";
      break;
    case Opcode::SERVER_COS_SIT_UP:
      std::cout << "SERVER_COS_SIT_UP\n";
      break;
    case Opcode::SERVER_ANIMATION_COS_REMOVE_MENU:
      std::cout << "SERVER_ANIMATION_COS_REMOVE_MENU\n";
      break;
    case Opcode::SERVER_COS_DELETE:
      std::cout << "SERVER_COS_DELETE\n";
      break;
    case Opcode::SERVER_ATTACK:
      std::cout << "SERVER_ATTACK\n";
      break;
    case Opcode::SERVER_SKILL_ATTACK:
      std::cout << "SERVER_SKILL_ATTACK\n";
      break;
    case Opcode::SERVER_END_SKILL:
      std::cout << "SERVER_END_SKILL\n";
      break;
    case Opcode::SERVER_BUFF_START:
      std::cout << "SERVER_BUFF_START\n";
      break;
    case Opcode::SERVER_BUFF_END:
      std::cout << "SERVER_BUFF_END\n";
      break;
    case Opcode::SERVER_DEAD:
      std::cout << "SERVER_DEAD\n";
      break;
    case Opcode::SERVER_DEAD2:
      std::cout << "SERVER_DEAD2\n";
      break;
    case Opcode::SERVER_PARTY_FORM:
      std::cout << "SERVER_PARTY_FORM\n";
      break;
    case Opcode::SERVER_PARTY_EDIT:
      std::cout << "SERVER_PARTY_EDIT\n";
      break;
    case Opcode::SERVER_PARTY_DELETE:
      std::cout << "SERVER_PARTY_DELETE\n";
      break;
    case Opcode::SERVER_PARTY_MATCHING:
      std::cout << "SERVER_PARTY_MATCHING\n";
      break;
    case Opcode::SERVER_PARTY_ACCEPT:
      std::cout << "SERVER_PARTY_ACCEPT\n";
      break;
    case Opcode::SERVER_PARTY_NEW_PARTY:
      std::cout << "SERVER_PARTY_NEW_PARTY\n";
      break;
    case Opcode::SERVER_PARTY_CHANGES:
      std::cout << "SERVER_PARTY_CHANGES\n";
      break;
    case Opcode::SERVER_PARTY_INVITE:
      std::cout << "SERVER_PARTY_INVITE\n";
      break;
    case Opcode::SERVER_OPEN_SHOP:
      std::cout << "SERVER_OPEN_SHOP\n";
      break;
    case Opcode::SERVER_CLOSE_SHOP:
      std::cout << "SERVER_CLOSE_SHOP\n";
      break;
    case Opcode::SERVER_SILK_AMOUNT:
      std::cout << "SERVER_SILK_AMOUNT\n";
      break;
    case Opcode::SERVER_TELEPORT:
      std::cout << "SERVER_TELEPORT\n";
      break;
    case Opcode::SERVER_ANIMATION_TELEPORT:
      std::cout << "SERVER_ANIMATION_TELEPORT\n";
      break;
    case Opcode::SERVER_STORAGE_GOLD:
      std::cout << "SERVER_STORAGE_GOLD\n";
      break;
    case Opcode::SERVER_STORAGE_ITEMS:
      std::cout << "SERVER_STORAGE_ITEMS\n";
      break;
    case Opcode::SERVER_STORAGE_END:
      std::cout << "SERVER_STORAGE_END\n";
      break;
    case Opcode::SERVER_ALCHEMY:
      std::cout << "SERVER_ALCHEMY\n";
      break;
    case Opcode::SERVER_ALCHEMYSTONE:
      std::cout << "SERVER_ALCHEMYSTONE\n";
      break;
    case Opcode::SERVER_REPAIR:
      std::cout << "SERVER_REPAIR\n";
      break;
    case Opcode::SERVER_ITEM_DURABILITY_CHANGE:
      std::cout << "SERVER_ITEM_DURABILITY_CHANGE\n";
      break;
    case Opcode::SERVER_CHARACTER_STUCK:
      std::cout << "SERVER_CHARACTER_STUCK\n";
      break;
    default:
      std::cout << "UNKNOWN\n";
      break;
  }
  return forwardPacket;
}