#include "Silkroad.h"
#include <windows.h>
#include "pk2Reader.h"
#include "common.h"
#include "BlowFish.h"
#include <algorithm>
using namespace edxLabs;

//-------------------------------------------------------------------------

// Parses a stream of data into a DivisionInfo object
DivisionInfo ParseDivisionInfo(void * buffer_, size_t size)
{
	DivisionInfo info;
	LPBYTE buffer = (LPBYTE)buffer_;
	DWORD index = 0;
	info.locale = buffer[index++];
	if(index > size) throw(std::exception("Invalid data format."));
	BYTE divCount = buffer[index++];
	if(index > size) throw(std::exception("Invalid data format."));
	for(BYTE x = 0; x < divCount; ++x)
	{
		Division tmpDiv;
		DWORD nameLength = *((LPDWORD)(buffer + index));
		index += 4;
		if(index > size) throw(std::exception("Invalid data format."));
		tmpDiv.name = (char *)(buffer + index);
		index += (nameLength + 1);
		if(index > size) throw(std::exception("Invalid data format."));
		BYTE ipCount = buffer[index++];
		if(index > size) throw(std::exception("Invalid data format."));
		for(BYTE y = 0; y < ipCount; ++y)
		{
			DWORD ipLength = *((LPDWORD)(buffer + index));
			index += 4;
			if(index > size) throw(std::exception("Invalid data format."));
			std::string ip = (char *)(buffer + index);
			tmpDiv.addresses.push_back(ip);
			index += (ipLength + 1);
			if(index > size) throw(std::exception("Invalid data format."));
		}
		info.divisions.push_back(tmpDiv);
	}
	return info;
}

//-------------------------------------------------------------------------

// Loads media.pk2 from the path at the index specified into a SilkroadData object
bool LoadPath(std::string path, SilkroadData & obj)
{
	pk2Reader reader;
	unsigned char keyData[] = {0x32, 0xCE, 0xDD, 0x7C, 0xBC, 0xA8};
	std::string s = path;
	s += "media.pk2";

	// Try to open the PK2
	if(reader.Open(s, keyData, 6) == false)
	{
		reader.Close();
		char er[1024] = {0};
		_snprintf(er, 1023, "The PK2Reader could not load %s", s.c_str());
		MessageBoxA(0, er, "Fatal Error", MB_ICONERROR);
		return false;
	}

	// Type.txt parser
	{
		std::list<pk2Entry> list1 = reader.GetEntry("type.txt");
		if(list1.empty())
		{
			char er[1024] = {0};
			_snprintf(er, 1023, "The PK2Reader could not find %s", "type.txt");
			MessageBoxA(0, er, "Fatal Error", MB_ICONERROR);
			return false;
		}
		memoryEntry me = reader.ExtractToMemory(*list1.begin());
		if(me.size == 0)
		{
			char er[1024] = {0};
			_snprintf(er, 1023, "The PK2Reader could not load %s", "type.txt");
			MessageBoxA(0, er, "Fatal Error", MB_ICONERROR);
			return false;
		}
		std::string buf = (char*)me.data;
		buf.resize(me.size);
		std::vector<std::string> tokens = TokenizeString(buf, "\n=");
		std::string key;
		std::string data;
		for(size_t x = 0; x < tokens.size(); ++x)
		{
			if((x+1) % 2 == 1)
				key = tokens[x];
			else
			{
				data = tokens[x];
				TrimString(data);
				TrimString(key);
				std::transform(key.begin(), key.end(), key.begin(), tolower);
				data.erase(data.begin());
				data.erase(data.end() - 1);
				obj.typeInfo[key] = data;
			}
		}
	}

	// SV.T parser
	{
		std::list<pk2Entry> list1 = reader.GetEntry("SV.T");
		if(list1.empty())
		{
			char er[1024] = {0};
			_snprintf(er, 1023, "The PK2Reader could not find %s", "SV.T");
			MessageBoxA(0, er, "Fatal Error", MB_ICONERROR);
			return false;
		}
		memoryEntry me = reader.ExtractToMemory(*list1.begin());
		if(me.size == 0)
		{
			char er[1024] = {0};
			_snprintf(er, 1023, "The PK2Reader could not load %s", "SV.T");
			MessageBoxA(0, er, "Fatal Error", MB_ICONERROR);
			return false;
		}
		BYTE * outBuffer = new BYTE[me.size];
		memcpy(outBuffer, me.data, me.size);
		DWORD inputSize = *((LPDWORD)(outBuffer));
		cBlowFish bf;
		bf.Initialize((LPBYTE)"SILKROADVERSION", 8);
		bf.Decode(outBuffer + 4, outBuffer + 4, inputSize);
		obj.version = atoi((char*)(outBuffer + 4));
		delete [] outBuffer;
	}

	// GATEPORT.TXT parser
	{
		std::list<pk2Entry> list1 = reader.GetEntry("GATEPORT.TXT");
		if(list1.empty())
		{
			char er[1024] = {0};
			_snprintf(er, 1023, "The PK2Reader could not find %s", "GATEPORT.TXT");
			MessageBoxA(0, er, "Fatal Error", MB_ICONERROR);
			return false;
		}
		memoryEntry me = reader.ExtractToMemory(*list1.begin());
		if(me.size == 0)
		{
			char er[1024] = {0};
			_snprintf(er, 1023, "The PK2Reader could not load %s", "GATEPORT.TXT");
			MessageBoxA(0, er, "Fatal Error", MB_ICONERROR);
			return false;
		}
		std::string buf = (char*)me.data;
		buf.resize(me.size);
		obj.gatePort = atoi(buf.c_str());
	}

	// DIVISIONINFO.TXT parser
	{
		std::list<pk2Entry> list1 = reader.GetEntry("DIVISIONINFO.TXT");
		if(list1.empty())
		{
			char er[1024] = {0};
			_snprintf(er, 1023, "The PK2Reader could not find %s", "DIVISIONINFO.TXT");
			MessageBoxA(0, er, "Fatal Error", MB_ICONERROR);
			return false;
		}
		memoryEntry me = reader.ExtractToMemory(*list1.begin());
		if(me.size == 0)
		{
			char er[1024] = {0};
			_snprintf(er, 1023, "The PK2Reader could not load %s", "DIVISIONINFO.TXT");
			MessageBoxA(0, er, "Fatal Error", MB_ICONERROR);
			return false;
		}
		try
		{
			obj.divInfo = ParseDivisionInfo(me.data, me.size);
		}
		catch(std::exception & e)
		{
			UNREFERENCED_PARAMETER(e);
			MessageBoxA(0, "There was an error parsing the divisions file.", "Fatal Error", MB_ICONERROR);
			return false;
		}
	}
	obj.path = path;
	reader.Close();
	return true;
}

//-------------------------------------------------------------------------

// Removes whitespace at the start and end of a string
void TrimString(std::string & source)
{
	std::string spaces = " \n\t\r";
	if(source.empty())
		return;
	size_t startSpace = source.find_first_of(spaces);
	size_t startNonSpace = source.find_first_not_of(spaces);
	while(startSpace != std::string::npos && startSpace < startNonSpace)
	{
		source.erase(startSpace, 1);
		startSpace = source.find_first_of(spaces);
		startNonSpace = source.find_first_not_of(spaces);
	}
	if(source.empty())
		return;
	startSpace = source.find_last_of(spaces);
	startNonSpace = source.find_last_not_of(spaces);
	while(startSpace != std::string::npos && startSpace > startNonSpace)
	{
		source.erase(startSpace, 1);
		startSpace = source.find_last_of(spaces);
		startNonSpace = source.find_last_not_of(spaces);
	}
}

//-------------------------------------------------------------------------
