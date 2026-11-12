# Genre glossary

Each section is one retrievable chunk. `maps_to` names the catalog genre used for a listener word that is not
in the catalog. `related` gives partial genre credit (0 to 1) to neighbouring catalog genres.

## pop
type: genre
synonyms: pop, top 40, k-pop, mainstream, radio hits
related: indie pop=0.6, edm=0.3
Catchy, polished songs built around a hook.

## indie pop
type: genre
synonyms: indie pop, indie, bedroom pop, dream pop
related: pop=0.6, folk=0.3
Pop songwriting with a lighter, homemade sound.

## lofi
type: genre
synonyms: lofi, lo-fi, lofi hip hop, lo-fi beats, chillhop, study beats
related: jazz=0.4, ambient=0.4, hip hop=0.3
Dusty drum loops, soft keys, and tape hiss. The default study soundtrack.

## rock
type: genre
synonyms: rock, punk, alternative, alt rock, grunge, guitar music
related: metal=0.4, indie pop=0.2
Guitar-driven songs with live drums.

## metal
type: genre
synonyms: metal, heavy metal, hardcore, metalcore, thrash
related: rock=0.5
The heaviest end of guitar music. Fast and distorted.

## edm
type: genre
synonyms: edm, electronic, house, techno, dance music, trance, rave music
related: pop=0.3, synthwave=0.4
Club-oriented electronic music with a steady four-on-the-floor beat.

## synthwave
type: genre
synonyms: synthwave, retrowave, outrun, 80s synth, 80s
related: edm=0.4, pop=0.2
Retro analog synths and drum machines with a night-drive feel.

## hip hop
type: genre
synonyms: hip hop, hip-hop, rap, trap, boom bap
related: r&b=0.5, lofi=0.2
Rhythm and rhymes over beats.

## r&b
type: genre
synonyms: r&b, rnb, soul, neo-soul, neo soul
related: hip hop=0.4, pop=0.3
Smooth vocals and groove.

## jazz
type: genre
synonyms: jazz, bebop, swing, smooth jazz
related: lofi=0.4, classical=0.2
Improvised, acoustic, and warm.

## bossa nova
type: genre
synonyms: bossa nova, bossa, samba jazz
maps_to: jazz
related: lofi=0.3
Not in the catalog. Bossa nova is Brazilian jazz, so jazz is the closest match, with lofi as a weak neighbour.

## classical
type: genre
synonyms: classical, orchestral, piano, symphony, string quartet
related: ambient=0.4, jazz=0.2
Composed acoustic music. Mostly instrumental.

## ambient
type: genre
synonyms: ambient, drone, soundscape, atmospheric music
related: classical=0.4, lofi=0.4
Texture over rhythm. Very low energy.

## folk
type: genre
synonyms: folk, singer-songwriter, acoustic folk, americana
related: country=0.5, indie pop=0.3
Acoustic guitar and voice.

## country
type: genre
synonyms: country, bluegrass, honky tonk
related: folk=0.5
Storytelling songs with twang.

## reggae
type: genre
synonyms: reggae, dancehall, ska, dub
related: hip hop=0.2, pop=0.2
Off-beat guitar and relaxed groove.
