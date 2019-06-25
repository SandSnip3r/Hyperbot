# PK2 Documentation

## Concepts

The Silkroad game client is always shipped with a few files with the .pk2 extenstion.
The Evolin private server comes with Data.pk2, Map.pk2, Media.pk2, Music.pk2, and Particles.pk2.
The headers are encrypted with a blowfish key.

## File contents

#### Data.pk2

- Navagation meshes _(.nvm)_
- Dungeons _(.dof)_
- Primitive information like textures, materials, mesh, animation, skeleton, and sounds _(combined in a .bsr resource file)_

#### Map.pk2

- The ground, water, and sky textures
- Terrain mesh information _(.m)_
- Terrain object placement information _(.02)_
- Terrain lighting information _(.t)_
 
#### Media.pk2

- Server dependency data
- Interface description files and textures

#### Music.pk2

- Music

#### Particles.pk2

- Particle description files _(.efp)_
- The particles' meshes _(.bms)_, textures _(.ddj)_, and animations _(.ban)_

## File structure

- Encrpted 256 byte header

## Existing parsing options

The C# source of AlchemyTool does some PK2 parsing on the media.pk2, Drew Benton's

## Resources

- [DaxterSoul's documentation](https://github.com/DummkopfOfHachtenduden/SilkroadDoc/wiki/JMXPACK)
- [Drew's PK2Tools 5-in-1 bundle](https://www.elitepvpers.com/forum/sro-hacks-bots-cheats-exploits/690658-pk2tools-5-1-bundle.html)
- [A PK2 extractor for language translation](https://www.elitepvpers.com/forum/sro-hacks-bots-cheats-exploits/364929-all-sro-pk2-extractor-english-patch-russian-chinese-korean-english.html)*

_*Unexplored_

## Keys

- default base key: 0x03, 0xF8, 0xE4, 0x44, 0x88, 0x99, 0x3F, 0x64, 0xFE, 0x35
- default ascii key: 169841
- byte[] bKey = new byte[] { 0x32, 0xCE, 0xDD, 0x7C, 0xBC, 0xA8 };*
- "\x32\x30\x30\x39\xC4\xEA" **

_*From AlchemyTool source, unknown reliability_    
_**pushedx's backup key from PK2Tools_

## Unorganized information

While the game client is running, it locks the pk2 files.
