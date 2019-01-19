// pk2Reader Lite

#ifndef _CRT_SECURE_NO_WARNINGS
	#define _CRT_SECURE_NO_WARNINGS
#endif

#include "pk2Reader.h"
#include <windows.h>
#include "BlowFish.h"
#include <queue>
#include <stack>
#include <list>
#include <set>
#include <sstream>
#include <algorithm>

//-------------------------------------------------------------------------

typedef int (*pk2EntryInternalFunc)(pk2Entry & entry, void * userData);

//-------------------------------------------------------------------------

#pragma pack(push, 1)
	struct pk2Header
	{
		char header[30];
		DWORD version;
		DWORD unk2;
		unsigned char reserved[218];
	};
	struct pk2EntryPrivate
	{
		BYTE type;
		char name[81];
		FILETIME accessTime;
		FILETIME createTime;
		FILETIME modifyTime;
		DWORD positionLow;
		DWORD positionHigh;
		DWORD size;
		DWORD nextChainLow;
		DWORD nextChainHigh;
		BYTE padding[2];
	};
	struct TUINT64
	{
		DWORD low;
		DWORD high;
		TUINT64() {low = 0; high = 0;}
		TUINT64(DWORD l, DWORD h) : low(l), high(h) { }
	};
#pragma pack(pop)

//-------------------------------------------------------------------------

struct pk2ReaderData
{
	// A handle to the file
	HANDLE hFile;

	// The file mapping object used to read from the file
	HANDLE hFileMapping;

	// A pointer to the bytes of the file
	PBYTE pMappedFileBase;

	// File size
	DWORD dwSizeLow;
	DWORD dwSizeHigh;

	// Our Blowfish decoding object
	cBlowFish blowFish;

	// Store a pointer to the first item
	TUINT64 rootEntryIndex;

	// Data structures we need for the parsing algorithm
	std::queue<TUINT64> bfsQueue;
	std::list<std::string> dfsPath;
	std::queue<std::string> bfsPath;
	std::string outputPath;

	// Track the entries extracted into memory to cleanup on application exit
	std::set<PBYTE> memoryListSet;

	// We need to store the system information for the dwAllocationGranularity field
	SYSTEM_INFO sysInfo;

	void ForEachPK2EntryDo_Reset();
	void ForEachPK2EntryDo_BFS(TUINT64 currentOffset, pk2EntryInternalFunc userFunc, void * userData);
	void ForEachPK2EntryDo_DFS(TUINT64 currentOffset, pk2EntryInternalFunc userFunc, void * userData);

	// Default ctor
	pk2ReaderData()
	{
		hFile = INVALID_HANDLE_VALUE;
		hFileMapping = 0;
		pMappedFileBase = 0;
		dwSizeLow = 0;
		dwSizeHigh = 0;
		rootEntryIndex.low = 0;
		rootEntryIndex.high = 0;
		GetSystemInfo(&sysInfo);
	}
};

//-------------------------------------------------------------------------

struct GetAllEntriesStruct
{
	pk2EntryUserFunc func;
	void * data;
	pk2Reader * reader;
	GetAllEntriesStruct(pk2EntryUserFunc f, void * d, pk2Reader * r)
		: func(f), data(d), reader(r) { }
};

//-------------------------------------------------------------------------

struct GetEntryStruct
{
	std::string name;
	std::list<pk2Entry> & results;
	GetEntryStruct(std::string n, std::list<pk2Entry> & r)
		: name(n), results(r) { }
};

//-------------------------------------------------------------------------

memoryEntry::memoryEntry()
{
	data = 0;
	size = 0;
	privatePtr = 0;
}

//-------------------------------------------------------------------------

memoryEntry::~memoryEntry()
{
}

//-------------------------------------------------------------------------

pk2Reader::pk2Reader()
{
	privateData = new pk2ReaderData;
}

//-------------------------------------------------------------------------

pk2Reader::~pk2Reader()
{
	// Cleanup the object if it is reused
	Close();
	delete privateData;
}

//-------------------------------------------------------------------------

// Opens the PK2 file with the specified Blowfish key
bool pk2Reader::Open(const std::string & filename, void * keyData, int keyCount)
{
	// Cleanup the object if it is reused
	Close();

	// Try to open the file
	privateData->hFile = CreateFileA(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if(privateData->hFile == INVALID_HANDLE_VALUE)
	{
		MessageBoxA(0, "Could not open the file.", "Error", MB_ICONERROR);
		return false;
	}

	// Get the file size
	privateData->dwSizeLow = GetFileSize(privateData->hFile, &privateData->dwSizeHigh);

	// Create a file mapping object of the file
	privateData->hFileMapping = CreateFileMapping(privateData->hFile, NULL, PAGE_READONLY, privateData->dwSizeHigh, privateData->dwSizeLow, NULL);
	if(privateData->hFileMapping == NULL)
	{
		Close();
		MessageBoxA(0, "Could not create a file mapping of the client.", "Error", MB_ICONERROR);
		return false;
	}

	// Map the view of the file
	privateData->pMappedFileBase = (PBYTE)MapViewOfFile(privateData->hFileMapping, FILE_MAP_READ, 0, 0, sizeof(pk2Header));
	if(privateData->pMappedFileBase == NULL)
	{
		Close();
		MessageBoxA(0, "Could not map a view of the file.", "Error", MB_ICONERROR);
		return false;
	}

	// Setup our blowfish object for decoding
	privateData->blowFish.Initialize((LPBYTE)keyData, keyCount);

	// Store a pointer to the pk2 header
	pk2Header * header = (pk2Header *)privateData->pMappedFileBase;
	if(header->version != 0x01000002)
	{
		Close();
		MessageBoxA(0, "header->version != 0x01000002", "Error", MB_ICONERROR);
		return false;
	}

	// Do some sanity checks
	std::string headerName = header->header;
	if(headerName != "JoyMax File Manager!\n")
	{
		Close();
		MessageBoxA(0, "headerName != \"JoyMax File Manager!\n\"", "Error", MB_ICONERROR);
		return false;
	}
	UnmapViewOfFile(privateData->pMappedFileBase);
	privateData->pMappedFileBase = 0;
	privateData->rootEntryIndex.low = sizeof(pk2Header);
	return true;
}

//-------------------------------------------------------------------------

// Closes the pk2Reader and frees all memory
void pk2Reader::Close()
{
	if(!privateData->memoryListSet.empty())
	{
		std::set<PBYTE>::iterator itr = privateData->memoryListSet.begin();
		while(itr != privateData->memoryListSet.end())
		{
			PBYTE ptr = (*itr);
			++itr;
			UnmapViewOfFile(ptr);
		}
		privateData->memoryListSet.clear();
	}

	if(privateData->pMappedFileBase)
	{
		UnmapViewOfFile(privateData->pMappedFileBase);
		privateData->pMappedFileBase = 0;
	}

	if(privateData->hFileMapping != NULL)
	{
		CloseHandle(privateData->hFileMapping);
		privateData->hFileMapping = NULL;
	}

	if(privateData->hFile != INVALID_HANDLE_VALUE)
	{
		CloseHandle(privateData->hFile);
		privateData->hFile = INVALID_HANDLE_VALUE;
	}

	privateData->dwSizeLow = 0;
	privateData->dwSizeHigh = 0;
}

//-------------------------------------------------------------------------

std::string PathFromList(std::list<std::string> & xfsPath)
{
	std::stringstream ss;
	std::list<std::string>::iterator itr = xfsPath.begin();
	while(itr != xfsPath.end())
	{
		ss << (*itr);
		++itr;
		if(itr != xfsPath.end())
			ss << "\\";
	}
	return ss.str();
}

//-------------------------------------------------------------------------

void pk2ReaderData::ForEachPK2EntryDo_BFS(TUINT64 currentOffset, pk2EntryInternalFunc userFunc, void * userData)
{
	pk2EntryPrivate * currentEntry = 0;
	pk2EntryPrivate tmpEntryPrivate;
	int entryIndexCtr = 0;
	std::string currentPath = "";
	if(bfsPath.empty()== false)
	{
		currentPath = bfsPath.front();
		bfsPath.pop();
	}
	int result = -1;
	DWORD dwLastHigh = -1;
	DWORD dwLastLow = -1;
	while(true)
	{
		pk2Entry tmpEntry;
		entryIndexCtr++;
		DWORD dwHigh = (currentOffset.high / sysInfo.dwAllocationGranularity) * sysInfo.dwAllocationGranularity;
		DWORD dwLow = (currentOffset.low / sysInfo.dwAllocationGranularity) * sysInfo.dwAllocationGranularity;
		if(dwLastHigh != dwHigh || dwLastLow != dwLow || pMappedFileBase == 0)
		{
			if(pMappedFileBase)
			{
				UnmapViewOfFile(pMappedFileBase);
			}
			// Store how many bytes we need to map in order to extract the entire file
			DWORD dwFinalCount = sysInfo.dwAllocationGranularity + sizeof(pk2EntryPrivate);

			// At this point, we need to make sure we don't over map into the file,
			// which would cause the MapViewOfFile function to fail.
			if(dwLow + dwFinalCount > dwSizeLow)
			{
				dwFinalCount = dwSizeLow - dwLow;
			}

			pMappedFileBase = (PBYTE)MapViewOfFile(hFileMapping, FILE_MAP_READ, dwHigh, dwLow, dwFinalCount);
			if(pMappedFileBase == 0) // Fatal error
			{
				printf("MapViewOfFile(%X, %i, %i, %i, %i) failed.\n", hFileMapping, FILE_MAP_READ, dwHigh, dwLow, sysInfo.dwAllocationGranularity + sizeof(pk2EntryPrivate));
				return;
			}
			dwLastHigh = dwHigh;
			dwLastLow = dwLow;
		}
		currentEntry = (pk2EntryPrivate *)(pMappedFileBase + currentOffset.low - dwLow);
		blowFish.Decode((BYTE *)currentEntry, (BYTE *)&tmpEntryPrivate, sizeof(pk2EntryPrivate));

		tmpEntry.type = tmpEntryPrivate.type;
		tmpEntry.name = tmpEntryPrivate.name;
		tmpEntry.path = currentPath;
		tmpEntry.position[0] = tmpEntryPrivate.positionLow;
		tmpEntry.position[1] = tmpEntryPrivate.positionHigh;
		tmpEntry.size = tmpEntryPrivate.size;
		tmpEntry.accessTime[0] = tmpEntryPrivate.accessTime.dwLowDateTime;
		tmpEntry.accessTime[1] = tmpEntryPrivate.accessTime.dwHighDateTime;
		tmpEntry.createTime[0] = tmpEntryPrivate.createTime.dwLowDateTime;
		tmpEntry.createTime[1] = tmpEntryPrivate.createTime.dwHighDateTime;
		tmpEntry.modifyTime[0] = tmpEntryPrivate.modifyTime.dwLowDateTime;
		tmpEntry.modifyTime[1] = tmpEntryPrivate.modifyTime.dwHighDateTime;
		result = userFunc(tmpEntry, userData);
		if(result == -1) // Stop entire search
			break;

		std::string entryName = tmpEntryPrivate.name;
		if(tmpEntryPrivate.type == 1 && entryName != "." && entryName != "..")
		{
			bfsQueue.push(TUINT64(tmpEntryPrivate.positionLow, tmpEntryPrivate.positionHigh));
			std::stringstream ss;
			if(currentPath.empty())
				ss << entryName;
			else
				ss << currentPath  << "\\" << entryName;
			bfsPath.push(ss.str());
		}
		if(tmpEntryPrivate.nextChainLow || tmpEntryPrivate.nextChainHigh)
		{
			currentOffset.low = tmpEntryPrivate.nextChainLow;
			currentOffset.high = tmpEntryPrivate.nextChainHigh;
			if(entryIndexCtr == 20)
			{
				entryIndexCtr = 0;
			}
		}
		else
		{
			// TODO: check for overflows
			currentOffset.low += sizeof(pk2EntryPrivate);
			if(entryIndexCtr == 20)
			{
				entryIndexCtr = 0;
				break;
			}
		}
		if(result == 0)// Stop current search
			break;
	}
	if(result == -1) // Stop entire search
	{
		return;
	}
	if(bfsQueue.empty() == false)
	{
		TUINT64 nextPosition = bfsQueue.front();
		bfsQueue.pop();
		ForEachPK2EntryDo_BFS(nextPosition, userFunc, userData);
	}
}

//-------------------------------------------------------------------------

void pk2ReaderData::ForEachPK2EntryDo_DFS(TUINT64 currentOffset, pk2EntryInternalFunc userFunc, void * userData)
{
	pk2EntryPrivate * currentEntry = 0;
	pk2EntryPrivate tmpEntryPrivate;
	int entryIndexCtr = 0;
	std::string currentPath = PathFromList(dfsPath);
	DWORD dwLastHigh = -1;
	DWORD dwLastLow = -1;
	while(true)
	{
		int result = 1;
		pk2Entry tmpEntry;
		entryIndexCtr++;
		DWORD dwHigh = (currentOffset.high / sysInfo.dwAllocationGranularity) * sysInfo.dwAllocationGranularity;
		DWORD dwLow = (currentOffset.low / sysInfo.dwAllocationGranularity) * sysInfo.dwAllocationGranularity;
		if(dwLastHigh != dwHigh || dwLastLow != dwLow || pMappedFileBase == 0)
		{
			if(pMappedFileBase)
			{
				UnmapViewOfFile(pMappedFileBase);
			}

			// Store how many bytes we need to map in order to extract the entire file
			DWORD dwFinalCount = sysInfo.dwAllocationGranularity + sizeof(pk2EntryPrivate);

			// At this point, we need to make sure we don't over map into the file,
			// which would cause the MapViewOfFile function to fail.
			if(dwLow + dwFinalCount > dwSizeLow)
			{
				dwFinalCount = dwSizeLow - dwLow;
			}

			pMappedFileBase = (PBYTE)MapViewOfFile(hFileMapping, FILE_MAP_READ, dwHigh, dwLow, dwFinalCount);
			if(pMappedFileBase == 0) // Fatal error
			{
				//MessageBox(0, "MapViewOfFile failed!", "Fatal Error", MB_ICONERROR);
				printf("MapViewOfFile(%X, %i, %i, %i, %i) failed for %s\n", hFileMapping, FILE_MAP_READ, dwHigh, dwLow, sysInfo.dwAllocationGranularity + sizeof(pk2EntryPrivate));
				return;
			}
			dwLastHigh = dwHigh;
			dwLastLow = dwLow;
		}
		currentEntry = (pk2EntryPrivate *)(pMappedFileBase + currentOffset.low - dwLow);
		blowFish.Decode((BYTE *)currentEntry, (BYTE *)&tmpEntryPrivate, sizeof(pk2EntryPrivate));

		tmpEntry.type = tmpEntryPrivate.type;
		tmpEntry.name = tmpEntryPrivate.name;
		tmpEntry.path = currentPath;
		tmpEntry.position[0] = tmpEntryPrivate.positionLow;
		tmpEntry.position[1] = tmpEntryPrivate.positionHigh;
		tmpEntry.size = tmpEntryPrivate.size;
		tmpEntry.accessTime[0] = tmpEntryPrivate.accessTime.dwLowDateTime;
		tmpEntry.accessTime[1] = tmpEntryPrivate.accessTime.dwHighDateTime;
		tmpEntry.createTime[0] = tmpEntryPrivate.createTime.dwLowDateTime;
		tmpEntry.createTime[1] = tmpEntryPrivate.createTime.dwHighDateTime;
		tmpEntry.modifyTime[0] = tmpEntryPrivate.modifyTime.dwLowDateTime;
		tmpEntry.modifyTime[1] = tmpEntryPrivate.modifyTime.dwHighDateTime;
		result = userFunc(tmpEntry, userData);
		if(result == -1)
			break;

		std::string entryName = tmpEntryPrivate.name;
		if(tmpEntryPrivate.type == 1 && entryName != "." && entryName != ".." && result == 1)
		{
			dfsPath.push_back(entryName);
			ForEachPK2EntryDo_DFS(TUINT64(tmpEntryPrivate.positionLow, tmpEntryPrivate.positionHigh), userFunc, userData);
			dfsPath.pop_back();
			dwLastHigh = -1;
			dwLastLow = -1;
		}
		if(tmpEntryPrivate.nextChainLow || tmpEntryPrivate.nextChainHigh)
		{
			currentOffset.low = tmpEntryPrivate.nextChainLow;
			currentOffset.high = tmpEntryPrivate.nextChainHigh;
			if(entryIndexCtr == 20)
			{
				entryIndexCtr = 0;
			}
		}
		else
		{
			// TODO: check for overflows
			currentOffset.low += sizeof(pk2EntryPrivate);
			if(entryIndexCtr == 20)
			{
				entryIndexCtr = 0;
				break;
			}
		}
	}
}

//-------------------------------------------------------------------------

void pk2ReaderData::ForEachPK2EntryDo_Reset()
{
	while(!bfsQueue.empty())
		bfsQueue.pop();
	dfsPath.clear();
	while(!bfsPath.empty())
		bfsPath.pop();
}

//-------------------------------------------------------------------------

int onGetAllEntries(pk2Entry & entry, void * userData)
{
	GetAllEntriesStruct * obj = (GetAllEntriesStruct *)userData;
	obj->func(obj->reader, entry, obj->data);
	return 1;
}

//-------------------------------------------------------------------------

// Processes all of the PK2 entries in a DFS manner and calls the user function
// with the user data passed in.
void pk2Reader::ForEachPK2EntryDo_DFS(pk2EntryUserFunc func, void * data)
{
	GetAllEntriesStruct obj(func, data, this);
	privateData->ForEachPK2EntryDo_Reset();
	privateData->ForEachPK2EntryDo_DFS(privateData->rootEntryIndex, onGetAllEntries, &obj);
	privateData->ForEachPK2EntryDo_Reset();
}

//-------------------------------------------------------------------------

// Processes all of the PK2 entries in a BFS manner and calls the user function
// with the user data passed in.
void pk2Reader::ForEachPK2EntryDo_BFS(pk2EntryUserFunc func, void * data)
{
	GetAllEntriesStruct obj(func, data, this);
	privateData->ForEachPK2EntryDo_Reset();
	privateData->ForEachPK2EntryDo_BFS(privateData->rootEntryIndex, onGetAllEntries, &obj);
	privateData->ForEachPK2EntryDo_Reset();
}

//-------------------------------------------------------------------------

// Returns a list of memoryEntry objects that contain pointers and sizes
// of the pk2Entry entries passed in the results object.
std::list<memoryEntry> pk2Reader::ExtractToMemory(std::list<pk2Entry> & results)
{
	std::list<memoryEntry> mem;
	std::list<pk2Entry>::iterator itr = results.begin();
	while(itr != results.end())
	{
		mem.push_back(ExtractToMemory(*itr));
		++itr;
	}
	return mem;
}

//-------------------------------------------------------------------------

// Returns a memoryEntry object that contains the pointer and size
// of the pk2Entry object result.
memoryEntry pk2Reader::ExtractToMemory(const pk2Entry & entry)
{
	memoryEntry mem;

	mem.size = entry.size;

	// Page aligned offsets for the view
	DWORD dwHigh = (entry.position[1] / privateData->sysInfo.dwAllocationGranularity) * privateData->sysInfo.dwAllocationGranularity;
	DWORD dwLow = (entry.position[0] / privateData->sysInfo.dwAllocationGranularity) * privateData->sysInfo.dwAllocationGranularity;

	// The size of the view we need at minimal to extract the entire file
	DWORD viewSize = entry.position[0] - dwLow + entry.size;

	// How many pages we need
	int pageCount = 1;

	// Calculate how many pages of dwAllocationGranularity we need for an
	// aligned allocation (more efficient).
	while(viewSize > privateData->sysInfo.dwAllocationGranularity)
	{
		viewSize -= privateData->sysInfo.dwAllocationGranularity;
		pageCount++;
	}

	// Store how many bytes we need to map in order to extract the entire file
	DWORD dwFinalCount = privateData->sysInfo.dwAllocationGranularity * pageCount;

	// At this point, we need to make sure we don't over map into the file,
	// which would cause the MapViewOfFile function to fail.
	if(dwLow + dwFinalCount > privateData->dwSizeLow)
	{
		dwFinalCount = privateData->dwSizeLow - dwLow;
	}

	// Create a memory aligned view of the file
	mem.privatePtr = (PBYTE)MapViewOfFile(privateData->hFileMapping, FILE_MAP_READ, dwHigh, dwLow, dwFinalCount);
	if(mem.privatePtr == 0) // Fatal error
	{
		printf("MapViewOfFile(%X, %i, %i, %i, %i) failed for %s\n", privateData->hFileMapping, FILE_MAP_READ, dwHigh, dwLow, dwFinalCount, entry.name.c_str());
		return memoryEntry();
	}

	// Store where the file begins at
	mem.data = mem.privatePtr + entry.position[0] - dwLow;

	// Manage the list internally so we don't leak memory
	privateData->memoryListSet.insert(mem.privatePtr);

	// Return the object
	return mem;
}

//-------------------------------------------------------------------------

// Frees a memoryEntry object
void pk2Reader::FreeMemoryEntry(memoryEntry & entry)
{
	PBYTE ptr = entry.privatePtr;
	if(privateData->memoryListSet.find(ptr) != privateData->memoryListSet.end())
		privateData->memoryListSet.erase(ptr);
	UnmapViewOfFile(ptr);
	entry = memoryEntry();
}

//-------------------------------------------------------------------------

// Frees a list of memoryEntry objects
void pk2Reader::FreeMemoryEntryList(std::list<memoryEntry> & entries)
{
	std::list<memoryEntry>::iterator itr = entries.begin();
	while(itr != entries.end())
	{
		PBYTE ptr = (*itr).privatePtr;
		++itr;
		if(privateData->memoryListSet.find(ptr) != privateData->memoryListSet.end())
			privateData->memoryListSet.erase(ptr);
		UnmapViewOfFile(ptr);
	}
	entries.clear();
}

//-------------------------------------------------------------------------

int onGetEntry(pk2Entry & entry, void * userData)
{
	GetEntryStruct * obj = (GetEntryStruct *)userData;
	if(entry.path.empty())
	{
		if(obj->name.find_first_of("\\") != std::string::npos)
			return 1;
		std::transform(entry.name.begin(), entry.name.end(), entry.name.begin(), tolower);
		if(obj->name == entry.name)
		{
			obj->results.push_back(entry);
			return -1;
		}
	}
	else
	{
		if(obj->name.find(entry.path) != 0)
			return 0;
		if(entry.type != 2)
			return 1;
		std::stringstream ss;
		ss << entry.path << "\\" << entry.name;
		std::string cmps = ss.str();
		std::transform(cmps.begin(), cmps.end(), cmps.begin(), tolower);
		if(cmps == obj->name)
		{
			obj->results.push_back(entry);
			return -1;
		}
	}
	return 1;
}

//-------------------------------------------------------------------------

// Returns the entry at the final path of name
std::list<pk2Entry> pk2Reader::GetEntry(std::string name)
{
	std::list<pk2Entry> results;
	std::transform(name.begin(), name.end(), name.begin(), tolower);
	GetEntryStruct obj(name, results);
	privateData->ForEachPK2EntryDo_Reset();
	privateData->ForEachPK2EntryDo_BFS(privateData->rootEntryIndex, onGetEntry, &obj);
	privateData->ForEachPK2EntryDo_Reset();
	return results;
}

//-------------------------------------------------------------------------
