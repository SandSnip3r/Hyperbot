#include "parsedPacket.hpp"

#include <iostream>

namespace packet::parsing {

//=========================================================================================================================================================

ParsedPacket::ParsedPacket(const PacketContainer &packet) : opcode_(static_cast<Opcode>(packet.opcode)) {}

Opcode ParsedPacket::opcode() const {
  return opcode_;
}

ParsedPacket::~ParsedPacket() {}

//=========================================================================================================================================================

ParsedUnknown::ParsedUnknown(const PacketContainer &packet) : ParsedPacket(packet) {}

//=========================================================================================================================================================

ParsedServerAgentCharacterData::ParsedServerAgentCharacterData(const PacketContainer &packet) : ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  uint32_t serverTime = stream.Read<uint32_t>();
  std::cout << "serverTime: " << serverTime << '\n';
  uint32_t refObjID = stream.Read<uint32_t>();
  std::cout << "refObjID: " << refObjID << '\n';
  uint8_t scale = stream.Read<uint8_t>();
  std::cout << "scale: " << (int)scale << '\n';
  uint8_t curLevel = stream.Read<uint8_t>();
  std::cout << "curLevel: " << (int)curLevel << '\n';
  uint8_t maxLevel = stream.Read<uint8_t>();
  std::cout << "maxLevel: " << (int)maxLevel << '\n';
  uint64_t expOffset = stream.Read<uint64_t>();
  std::cout << "expOffset: " << expOffset << '\n';
  uint32_t sExpOffset = stream.Read<uint32_t>();
  std::cout << "sExpOffset: " << sExpOffset << '\n';
  uint64_t remainGold = stream.Read<uint64_t>();
  std::cout << "remainGold: " << remainGold << '\n';
  uint32_t remainSkillPoint = stream.Read<uint32_t>();
  std::cout << "remainSkillPoint: " << remainSkillPoint << '\n';
  uint16_t remainStatPoint = stream.Read<uint16_t>();
  std::cout << "remainStatPoint: " << remainStatPoint << '\n';
  uint8_t remainHwanCount = stream.Read<uint8_t>();
  std::cout << "remainHwanCount: " << (int)remainHwanCount << '\n';
  uint32_t gatheredExpPoint = stream.Read<uint32_t>();
  std::cout << "gatheredExpPoint: " << gatheredExpPoint << '\n';
  uint32_t hP = stream.Read<uint32_t>();
  std::cout << "hP: " << hP << '\n';
  uint32_t mP = stream.Read<uint32_t>();
  std::cout << "mP: " << mP << '\n';
  uint8_t autoInverstExp = stream.Read<uint8_t>();
  std::cout << "autoInverstExp: " << (int)autoInverstExp << '\n';
  uint8_t dailyPK = stream.Read<uint8_t>();
  std::cout << "dailyPK: " << (int)dailyPK << '\n';
  uint16_t totalPK = stream.Read<uint16_t>();
  std::cout << "totalPK: " << totalPK << '\n';
  uint32_t pKPenaltyPoint = stream.Read<uint32_t>();
  std::cout << "pKPenaltyPoint: " << pKPenaltyPoint << '\n';
  uint8_t hwanLevel = stream.Read<uint8_t>();
  std::cout << "hwanLevel: " << (int)hwanLevel << '\n';
  uint8_t freePVP = stream.Read<uint8_t>(); //0 = None, 1 = Red, 2 = Gray, 3 = Blue, 4 = White, 5 = Gold
  std::cout << "freePVP: " << (int)freePVP << '\n';

  // //Inventory
  uint8_t inventorySize = stream.Read<uint8_t>();
  std::cout << "inventorySize: " << (int)inventorySize << '\n';
  uint8_t inventoryItemCount = stream.Read<uint8_t>();
  std::cout << "inventoryItemCount: " << (int)inventoryItemCount << '\n';
  // for (int i = 0; i < Inventory.ItemCount; i++)
  // {
  //     1   byte    item.Slot
      
  //     4   uint    item.RentType
  //     if(item.RentType == 1)    
  //     {
  //         2   ushort  item.RentInfo.CanDelete
  //         4   uint    item.RentInfo.PeriodBeginTime
  //         4   uint    item.RentInfo.PeriodEndTime        
  //     }
  //     else if(item.RentType == 2)
  //     {
  //         2   ushort  item.RentInfo.CanDelete
  //         2   ushort  item.RentInfo.CanRecharge
  //         4   uint    item.RentInfo.MeterRateTime        
  //     }
  //     else if(item.RentType == 3)
  //     {
  //         2   ushort  item.RentInfo.CanDelete
  //         2   ushort  item.RentInfo.CanRecharge
  //         4   uint    item.RentInfo.PeriodBeginTime
  //         4   uint    item.RentInfo.PeriodEndTime   
  //         4   uint    item.RentInfo.PackingTime        
  //     }
      
  //     4   uint    item.RefItemID
  //     if(item.TypeID1 == 3)
  //     {
  //         //ITEM_        
  //         if(item.TypeID2 == 1)
  //         {
  //             //ITEM_CH
  //             //ITEM_EU
  //             //AVATAR_
  //             1   byte    item.OptLevel
  //             8   ulong   item.Variance
  //             4   uint    item.Data       //Durability
  //             1   byte    item.MagParamNum
  //             for(int paramIndex = 0; paramIndex < item.MagParamNum; paramIndex++)
  //             {
  //                 4   uint    magParam.Type
  //                 4   uint    magParam.Value                
  //             }
              
  //             1   byte    bindingOptionType   //1 = Socket
  //             1   byte    bindingOptionCount
  //             for (int bindingOptionIndex = 0; bindingOptionIndex < bindingOptionCount; bindingOptionIndex++)
  //             {
  //                 1   byte bindingOption.Slot
  //                 4   uint bindingOption.ID
  //                 4   uint bindingOption.nParam1
  //             }
              
  //             1   byte    bindingOptionType   //2 = Advanced elixir
  //             1   byte    bindingOptionCount
  //             for (int bindingOptionIndex = 0; bindingOptionIndex < bindingOptionCount; bindingOptionIndex++)
  //             {
  //                 1   byte bindingOption.Slot
  //                 4   uint bindingOption.ID
  //                 4   uint bindingOption.OptValue
  //             }            
  //         }
  //         else if(item.TypeID2 == 2)
  //         {            
  //             if(item.TypeID3 == 1)
  //             {                                
  //                 //ITEM_COS_P
  //                 1   byte    State
  //                 4   uint    RefObjID
  //                 2   ushort  Name.Length
  //                 *   string  Name
  //                 if(item.TypeID4 == 2)
  //                 {
  //                     //ITEM_COS_P (Ability)
  //                     4   uint    SecondsToRentEndTime
  //                 }
  //                 1   byte    unkByte0
  //             }
  //             else if(item.TypeID3 == 2)
  //             {
  //                 //ITEM_ETC_TRANS_MONSTER
  //                 4   uint    RefObjID
  //             }
  //             else if(item.TypeID3 == 3)
  //             {
  //                 //MAGIC_CUBE
  //                 4   uint    Quantity        //Do not confuse with StackCount, this indicates the amount of elixirs in the cube
  //             }
  //         }
  //         else if(item.TypeID2 == 3)
  //         {
  //             //ITEM_ETC
  //             2   ushort  item.StackCount
              
  //             if(item.TypeID3 == 11)
  //             {
  //                 if(item.TypeID4 == 1 || item.TypeID4 == 2)
  //                 {
  //                     //MAGICSTONE, ATTRSTONE
  //                     1   byte    AttributeAssimilationProbability
  //                 }
  //             }
  //             else if(item.TypeID3 == 14 || item.TypeID4 == 2)
  //             {
  //                 //ITEM_MALL_GACHA_CARD_WIN
  //                 //ITEM_MALL_GACHA_CARD_LOSE
  //                 1   byte    item.MagParamCount
  //                 for (int paramIndex = 0; paramIndex < MagParamNum; paramIndex++)
  //                 {
  //                     4   uint magParam.Type
  //                     4   uint magParam.Value
  //                 }
  //             }
  //         } 
  //     }
  // }

  // //AvatarInventory
  // 1   byte    AvatarInventory.Size
  // 1   byte    AvatarInventory.ItemCount
  // for (int i = 0; i < Inventory.ItemCount; i++)
  // {
  //     1   byte    item.Slot
      
  //     4   uint    item.RentType
  //     if(item.RentType == 1)    
  //     {
  //         2   ushort  item.RentInfo.CanDelete
  //         4   uint    item.RentInfo.PeriodBeginTime
  //         4   uint    item.RentInfo.PeriodEndTime        
  //     }
  //     else if(item.RentType == 2)
  //     {
  //         2   ushort  item.RentInfo.CanDelete
  //         2   ushort  item.RentInfo.CanRecharge
  //         4   uint    item.RentInfo.MeterRateTime        
  //     }
  //     else if(item.RentType == 3)
  //     {
  //         2   ushort  item.RentInfo.CanDelete
  //         2   ushort  item.RentInfo.CanRecharge
  //         4   uint    item.RentInfo.PeriodBeginTime
  //         4   uint    item.RentInfo.PeriodEndTime   
  //         4   uint    item.RentInfo.PackingTime        
  //     }
      
  //     4   uint    item.RefItemID
  //     if(item.TypeID1 == 3)
  //     {
  //         //ITEM_        
  //         if(item.TypeID2 == 1) //TODO: Narrow filters for AvatarInventory
  //         {
  //             //ITEM_CH
  //             //ITEM_EU
  //             //AVATAR_
  //             1   byte    item.OptLevel
  //             8   ulong   item.Variance
  //             4   uint    item.Data       //Durability
  //             1   byte    item.MagParamNum
  //             for(int paramIndex = 0; paramIndex < item.MagParamNum; paramIndex++)
  //             {
  //                 4   uint    magParam.Type
  //                 4   uint    magParam.Value                
  //             }
              
  //             1   byte    bindingOptionType   //1 = Socket
  //             1   byte    bindingOptionCount
  //             for (int bindingOptionIndex = 0; bindingOptionIndex < bindingOptionCount; bindingOptionIndex++)
  //             {
  //                 1   byte bindingOption.Slot
  //                 4   uint bindingOption.ID
  //                 4   uint bindingOption.nParam1
  //             }
              
  //             1   byte    bindingOptionType   //2 = Advanced elixir
  //             1   byte    bindingOptionCount
  //             for (int bindingOptionIndex = 0; bindingOptionIndex < bindingOptionCount; bindingOptionIndex++)
  //             {
  //                 1   byte bindingOption.Slot
  //                 4   uint bindingOption.ID
  //                 4   uint bindingOption.OptValue
  //             }            
  //         }
  //     }
  // }

  // 1   byte    unkByte1    //not a counter

  // //Masteries
  // 1   byte    nextMastery
  // while(nextMastery == 1)
  // {
  //     4   uint    mastery.ID
  //     1   byte    mastery.Level   
      
  //     1   byte    nextMastery
  // }

  // 1   byte    unkByte2    //not a counter

  // //Skills
  // 1   byte    nextSkill
  // while(nextSkill == 1)
  // {
  //     4   uint    skill.ID
  //     1   byte    skill.Enabled   
      
  //     1   byte    nextSkill
  // }

  // //Quests
  // 2   ushort  CompletedQuestCount
  // *   uint[]  CompletedQuests

  // 1   byte    ActiveQuestCount
  // for(int activeQuestIndex = 0; activeQuestIndex < ActiveQuestCount; activeQuestIndex++)
  // {
  //     4   uint    quest.RefQuestID
  //     1   byte    quest.AchivementCount
  //     1   byte    quest.RequiresAutoShareParty
  //     1   byte    quest.Type
  //     if(quest.Type == 28)
  //     {
  //         4   uint    remainingTime
  //     }
  //     1   byte    quest.Status
      
  //     if(quest.Type != 8)
  //     {
  //         1   byte    quest.ObjectiveCount
  //         for(int objectiveIndex = 0; objectiveIndex < quest.ObjectiveCount; objectiveIndex++)
  //         {
  //             1   byte    objective.ID
  //             1   byte    objective.Status        //0 = Done, 1  = On
  //             2   ushort  objective.Name.Length
  //             *   string  objective.Name
  //             1   byte    objective.TaskCount
  //             for(int taskIndex = 0; taskIndex < objective.TaskCount; taskIndex++)
  //             {
  //                 4   uint    task.Value
  //             }
  //         }
  //     }
      
  //     if(quest.Type == 88)
  //     {
  //         1   byte    RefObjCount
  //         for(int refObjIndex = 0; refObjIndex < RefObjCount; refObjIndex++)
  //         {
  //             4   uint    RefObjID    //NPCs
  //         }
  //     }
  // }

  // 1   byte    unkByte3        //Structure changes!!!

  // //CollectionBook
  // 4   uint    CollectionBookStartedThemeCount
  // for(int i = 0; i < StartedCollectionCount)
  // {
  //     4   uint    theme.Index
  //     4   uint    theme.StartedDateTime   //SROTimeStamp
  //     4   uint    theme.Pages
  // }

  // 4   uint    UniqueID

  // //Position
  // 2   ushort  Position.RegionID
  // 4   float   Position.X
  // 4   float   Position.Y
  // 4   float   Position.Z
  // 2   ushort  Position.Angle

  // //Movement
  // 1   byte    Movement.HasDestination
  // 1   byte    Movement.Type
  // if(Movement.HasDestination)
  // {    
  //     2   ushort  Movement.DestinationRegion        
  //     if(Position.RegionID < short.MaxValue)
  //     {
  //         //World
  //         2   ushort  Movement.DestinationOffsetX
  //         2   ushort  Movement.DestinationOffsetY
  //         2   ushort  Movement.DestinationOffsetZ
  //     }
  //     else
  //     {
  //         //Dungeon
  //         4   uint  Movement.DestinationOffsetX
  //         4   uint  Movement.DestinationOffsetY
  //         4   uint  Movement.DestinationOffsetZ
  //     }
  // }
  // else
  // {
  //     1   byte    Movement.Source     //0 = Spinning, 1 = Sky-/Key-walking
  //     2   ushort  Movement.Angle      //Represents the new angle, character is looking at
  // }

  // //State
  // 1   byte    State.LifeState         //1 = Alive, 2 = Dead
  // 1   byte    State.unkByte0
  // 1   byte    State.MotionState       //0 = None, 2 = Walking, 3 = Running, 4 = Sitting
  // 1   byte    State.Status            //0 = None, 1 = Hwan, 2 = Untouchable, 3 = GameMasterInvincible, 5 = GameMasterInvisible, 5 = ?, 6 = Stealth, 7 = Invisible
  // 4   float   State.WalkSpeed
  // 4   float   State.RunSpeed
  // 4   float   State.HwanSpeed
  // 1   byte    State.BuffCount
  // for (int i = 0; i < State.BuffCount; i++)
  // {
  //     4   uint    Buff.RefSkillID
  //     4   uint    Buff.Duration
  //     if(skill.Params.Contains(1701213281))
  //     {
  //         //1701213281 -> atfe -> "auto transfer effect" like Recovery Division
  //         1   bool    IsCreator
  //     }
  // }

  // 2   ushort  Name.Length
  // *   string  Name
  // 2   ushort  JobName.Length
  // *   string  JobName
  // 1   byte    JobType
  // 1   byte    JobLevel
  // 4   uint    JobExp
  // 4   uint    JobContribution
  // 4   uint    JobReward
  // 1   byte    PVPState                //0 = White, 1 = Purple, 2 = Red
  // 1   byte    TransportFlag
  // 1   byte    InCombat
  // if(TransportFlag == 1)
  // {
  //     4   uint    Transport.UniqueID
  // }

  // 1   byte    PVPFlag                 //0 = Red Side, 1 = Blue Side, 0xFF = None
  // 8   ulong   GuideFlag
  // 4   uint    JID
  // 1   byte    GMFlag

  // 1   byte    ActivationFlag          //ConfigType:0 --> (0 = Not activated, 7 = activated)
  // 1   byte    Hotkeys.Count           //ConfigType:1
  // for(int i = 0; i < hotkeyCount; i++)
  // {
  //     1   byte    hotkey.SlotSeq
  //     1   byte    hotkey.SlotContentType
  //     4   uint    hotkey.SlotData
  // }
  // 2   ushort  AutoHPConfig            //ConfigType:11
  // 2   ushort  AutoMPConfig            //ConfigType:12
  // 2   ushort  AutoUniversalConfig     //ConfigType:13
  // 1   byte    AutoPotionDelay         //ConfigType:14

  // 1   byte    blockedWhisperCount
  // for(int i = 0; i < blockedWhisperCount; i++)
  // {
  //     2   ushort  Target.Length
  //     *   string  Target
  // }

  // 4   uint    unkUShort0      //Structure changes!!!
  // 1   byte    unkByte4        //Structure changes!!!
}

//=========================================================================================================================================================

ParsedServerAgentCharacterSelectionJoinResponse::ParsedServerAgentCharacterSelectionJoinResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  result_ = stream.Read<uint8_t>();
  if (result_ == 0x02) {
    errorCode_ = stream.Read<uint16_t>();
  }
}

uint8_t ParsedServerAgentCharacterSelectionJoinResponse::result() const {
  return result_;
}

uint16_t ParsedServerAgentCharacterSelectionJoinResponse::errorCode() const {
  return errorCode_;
}

//=========================================================================================================================================================

ParsedServerAgentCharacterSelectionActionResponse::ParsedServerAgentCharacterSelectionActionResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  action_ = static_cast<PacketEnums::CharacterSelectionAction>(stream.Read<uint8_t>());
  result_ = stream.Read<uint8_t>();
  if (result_ == 0x01 && action_ == PacketEnums::CharacterSelectionAction::kList) {
    // Listing characters
    const uint8_t kCharCount = stream.Read<uint8_t>();
    for (int i=0; i<kCharCount; ++i) {
      PacketInnerStructures::CharacterSelection::Character character;
      character.refObjID = stream.Read<uint32_t>();
      const uint16_t kNameLength = stream.Read<uint16_t>();
      character.name = stream.Read_Ascii(kNameLength);
      character.scale = stream.Read<uint8_t>();
      character.curLevel = stream.Read<uint8_t>();
      character.expOffset = stream.Read<uint64_t>();
      character.strength = stream.Read<uint16_t>();
      character.intelligence = stream.Read<uint16_t>();
      character.statPoint = stream.Read<uint16_t>();
      character.curHP = stream.Read<uint32_t>();
      character.curMP = stream.Read<uint32_t>();
      character.isDeleting = stream.Read<bool>();
      if (character.isDeleting) {
        character.charDeleteTime = stream.Read<uint32_t>();
      }
      character.guildMemberClass = stream.Read<uint8_t>();
      character.isGuildRenameRequired = stream.Read<bool>();
      if (character.isGuildRenameRequired) {
        const uint16_t kCurrentGuildNameLength = stream.Read<uint16_t>();
        character.currentGuildName = stream.Read_Ascii(kCurrentGuildNameLength);
      }
      character.academyMemberClass = stream.Read<uint8_t>();
      const uint8_t kItemCount = stream.Read<uint8_t>();
      for (int j=0; j<kItemCount; ++j) {
        PacketInnerStructures::CharacterSelection::Item item;   
        item.refId = stream.Read<uint32_t>();
        item.plus = stream.Read<uint8_t>();
        character.items.emplace_back(std::move(item));
      }
      const uint8_t kAvatarCount = stream.Read<uint8_t>();
      for (int j=0; j<kAvatarCount; ++j) {
        PacketInnerStructures::CharacterSelection::Avatar avatar;   
        avatar.refId = stream.Read<uint32_t>();
        avatar.plus = stream.Read<uint8_t>();
        character.avatars.emplace_back(std::move(avatar));
      }
      characters_.emplace_back(std::move(character));
    }
  } else if (result_ == 0x02) {
    errorCode_ = stream.Read<uint16_t>();
  }
}

PacketEnums::CharacterSelectionAction ParsedServerAgentCharacterSelectionActionResponse::action() const {
  return action_;
}

uint8_t ParsedServerAgentCharacterSelectionActionResponse::result() const {
  return result_;
}

const std::vector<PacketInnerStructures::CharacterSelection::Character>& ParsedServerAgentCharacterSelectionActionResponse::characters() const {
  return characters_;
}

uint16_t ParsedServerAgentCharacterSelectionActionResponse::errorCode() const {
  return errorCode_;
}

//=========================================================================================================================================================

ParsedServerAuthResponse::ParsedServerAuthResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  result_ = stream.Read<uint8_t>();
  if (result_ == 0x02) {
    errorCode_ = stream.Read<uint8_t>();
  }
}

uint8_t ParsedServerAuthResponse::result() const {
  return result_;
}

uint8_t ParsedServerAuthResponse::errorCode() const {
  return errorCode_;
}

//=========================================================================================================================================================

ParsedLoginClientInfo::ParsedLoginClientInfo(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  uint16_t serviceNameLength = stream.Read<uint16_t>();
  serviceName_ = stream.Read_Ascii(serviceNameLength);
  /* uint8_t isLocal = */ stream.Read<uint8_t>();
}

std::string ParsedLoginClientInfo::serviceName() const {
  return serviceName_;
}

//=========================================================================================================================================================

ParsedLoginResponse::ParsedLoginResponse(const PacketContainer &packet) :
      ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  result_ = static_cast<PacketEnums::LoginResult>(stream.Read<uint8_t>());
  if (result_ == PacketEnums::LoginResult::kSuccess) {
    token_ = stream.Read<uint32_t>();
    uint16_t ipLength = stream.Read<uint16_t>();
    std::string ip = stream.Read_Ascii(ipLength);
    uint16_t port = stream.Read<uint16_t>();
  } else if (result_ == PacketEnums::LoginResult::kFailed) {
    uint8_t errorCode = stream.Read<uint8_t>();
    if (errorCode == 0x01) {
      uint32_t maxAttempts = stream.Read<uint32_t>();
      uint32_t currentAttempts = stream.Read<uint32_t>();
    } else if (errorCode == 0x02) {
      PacketEnums::LoginBlockType blockType = static_cast<PacketEnums::LoginBlockType>(stream.Read<uint8_t>());
      if (blockType == PacketEnums::LoginBlockType::kPunishment) {
        uint16_t reasonLength = stream.Read<uint16_t>();
        std::string reason = stream.Read_Ascii(reasonLength);
        uint16_t endDateYear = stream.Read<uint16_t>();
        uint16_t endDateMonth = stream.Read<uint16_t>();
        uint16_t endDateDay = stream.Read<uint16_t>();
        uint16_t endDateHour = stream.Read<uint16_t>();
        uint16_t endDateMinute = stream.Read<uint16_t>();
        uint16_t endDateSecond = stream.Read<uint16_t>();
        uint16_t endDateMicrosecond = stream.Read<uint16_t>();
      }
    }
  } else if (result_ == PacketEnums::LoginResult::kOther) {
    /* uint8_t unkByte0 = */ stream.Read<uint8_t>();
    /* uint8_t unkByte1 = */ stream.Read<uint8_t>();
    uint16_t messageLength = stream.Read<uint16_t>();
    /* std::string message = */ stream.Read_Ascii(messageLength);
    /* uint16_t unkUShort0 = */ stream.Read<uint16_t>();
  }
}

PacketEnums::LoginResult ParsedLoginResponse::result() const {
  return result_;
}

uint32_t ParsedLoginResponse::token() const {
  return token_;
}


//=========================================================================================================================================================

ParsedLoginServerList::ParsedLoginServerList(const PacketContainer &packet) :
    ParsedPacket(packet) {
  StreamUtility stream = packet.data;
  uint8_t globalOpFlag = stream.Read<uint8_t>();
  while (globalOpFlag == 0x01) {
    // Read a "global op" , will be something like "SRO_Vietnam_TestLocal"
    uint8_t globalOpType = stream.Read<uint8_t>(); // For Atomix, its SRO_Taiwan_TestIn
    uint16_t globalNameLength = stream.Read<uint16_t>();
    std::string globalName = stream.Read_Ascii(globalNameLength);
    globalOpFlag = stream.Read<uint8_t>();
  }
  uint8_t shardFlag = stream.Read<uint8_t>();
  while (shardFlag == 0x01) {
    // Read a "shard" , will be something like "Atomix"
    shardId_ = stream.Read<uint16_t>();
    uint16_t shardNameLength = stream.Read<uint16_t>();
    std::string shardName = stream.Read_Ascii(shardNameLength);
    uint16_t shardCurrent = stream.Read<uint16_t>();
    uint16_t shardCapacity = stream.Read<uint16_t>();
    bool shardOnline = stream.Read<uint8_t>();
    uint8_t globalOpId = stream.Read<uint8_t>(); // Idk what this is, i guess globalOpType from above
    shardFlag = stream.Read<uint8_t>();
  }
}

uint16_t ParsedLoginServerList::shardId() const {
  return shardId_;
}

//=========================================================================================================================================================

ParsedClientCafe::ParsedClientCafe(const PacketContainer &packet) :
      ParsedPacket(packet) {
  //
}

//=========================================================================================================================================================

ParsedClientChat::ParsedClientChat(const PacketContainer &packet) :
      ParsedPacket(packet) {
  // 1   byte    chatType
  // 1   byte    chatIndex
  // if(chatType == ChatType.PM)
  // {
  //     2   ushort  reciver.Length
  //     *   string  reciver
  // }
  // 2   ushort  message.Length
  // *   string  message
  StreamUtility stream = packet.data;
  chatType_ = static_cast<PacketEnums::ChatType>(stream.Read<uint8_t>());
  chatIndex_ = stream.Read<uint8_t>();
  if (chatType_ == PacketEnums::ChatType::kPm) {
    const uint16_t kReceiverNameLength = stream.Read<uint16_t>();
    receiverName_ = stream.Read_Ascii(kReceiverNameLength);
  }
  const uint16_t kMessageLength = stream.Read<uint16_t>();
  message_ = stream.Read_Ascii(kMessageLength);
}

PacketEnums::ChatType ParsedClientChat::chatType() const {
  return chatType_;
}

uint8_t ParsedClientChat::chatIndex() const {
  return chatIndex_;
}

const std::string& ParsedClientChat::receiverName() const {
  return receiverName_;
}

const std::string& ParsedClientChat::message() const {
  return message_;
}

//=========================================================================================================================================================
} // namespace packet::parsing