#ifndef BLOWFISH_H_
#define BLOWFISH_H_

#ifndef _WINDOWS_
	#include <windows.h>
#endif

/*
	Blowfish
	By: Jim Conger (original Bruce Schneier)
	Url: http://www.schneier.com/blowfish-download.html
*/

class cBlowFish
{
private:
	DWORD 		* PArray;
	DWORD		(* SBoxes)[256];
	void 		Blowfish_encipher(DWORD *xl, DWORD *xr);
	void 		Blowfish_decipher(DWORD *xl, DWORD *xr);

public:
	cBlowFish();
	~cBlowFish();
	void 		Initialize(BYTE key[], int keybytes);
	DWORD		GetOutputLength(DWORD lInputLong);
	DWORD		Encode(BYTE * pInput, BYTE * pOutput, DWORD lSize);
	void		Decode(BYTE * pInput, BYTE * pOutput, DWORD lSize);
};

#endif