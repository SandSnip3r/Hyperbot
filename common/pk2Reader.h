// pk2Reader Lite

#pragma once

#ifndef PK2READER_H_
#define PK2READER_H_

//-------------------------------------------------------------------------

#ifndef _STRING_
	#include <string>
#endif

#ifndef _VECTOR_
	#include <vector>
#endif

#ifndef _LIST_
	#include <list>
#endif

//-------------------------------------------------------------------------

struct pk2Entry
{
	unsigned char type;
	unsigned long size;
	unsigned long position[2];
	unsigned long accessTime[2];
	unsigned long createTime[2];
	unsigned long modifyTime[2];
	std::string name;
	std::string path;
};

struct memoryEntry
{
	unsigned char * data;
	size_t size;
	unsigned char * privatePtr;
	memoryEntry();
	~memoryEntry();
};

//-------------------------------------------------------------------------

class pk2Reader;
typedef void (*pk2EntryUserFunc)(pk2Reader * reader, pk2Entry & entry, void * userData);

// Forward declaration
struct pk2ReaderData;

// This is a PK2 reading class. It will load a pk2, parse through the encrypted 
// file entries and allow a user to extract files.
class pk2Reader
{
private:
	pk2ReaderData * privateData;

public:
	pk2Reader();

	// Default dtor, cleans up the class on destruction
	~pk2Reader();

	// Opens the PK2 file with the specified Blowfish key
	bool Open(const std::string & filename, void * keyData, int keyCount);

	// Closes the pk2Reader and frees all memory
	void Close();

	// Returns a list of memoryEntry objects that contain pointers and sizes
	// of the pk2Entry entries passed in the results object.
	std::list<memoryEntry> ExtractToMemory(std::list<pk2Entry> & results);

	// Returns a memoryEntry object that contains the pointer and size
	// of the pk2Entry object result.
	memoryEntry ExtractToMemory(const pk2Entry & result);

	// Frees a list of memoryEntry objects
	void FreeMemoryEntryList(std::list<memoryEntry> & entries);

	// Frees a memoryEntry object
	void FreeMemoryEntry(memoryEntry & entry);

	// Returns the entry at the final path of name
	std::list<pk2Entry> GetEntry(std::string name);

	// Processes all of the PK2 entries in a BFS manner and calls the user function
	// with the user data passed in.
	void ForEachPK2EntryDo_BFS(pk2EntryUserFunc func, void * data);

	// Processes all of the PK2 entries in a DFS manner and calls the user function
	// with the user data passed in.
	void ForEachPK2EntryDo_DFS(pk2EntryUserFunc func, void * data);
};

//-------------------------------------------------------------------------

#endif
