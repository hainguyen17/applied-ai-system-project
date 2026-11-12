# Mood glossary

Each section is one retrievable chunk. Metadata lines (`key: value`) come first, then a short description.
`maps_to` names the catalog mood to use when the listener's word is not a catalog mood.
`related` gives partial credit (0 to 1) to neighbouring catalog moods.
`energy` is the typical energy range, used when the listener does not say how energetic.

## happy
type: mood
energy: 0.65-0.85
synonyms: happy, cheerful, joyful, good mood, feel good, feel-good, sunny, cheery
related: uplifting=0.6, euphoric=0.4
Bright, major-key songs with high valence. Good for mornings, chores, and getting out of a slump.

## chill
type: mood
energy: 0.30-0.50
synonyms: chill, mellow, laid-back, laid back, low-key, easygoing, unwind
related: relaxed=0.7, peaceful=0.5
Unhurried songs that sit in the background. Lower energy, soft drums, little tension.

## relaxed
type: mood
energy: 0.30-0.50
synonyms: relaxed, relax, relaxing, lounge, lazy, slow sunday
related: chill=0.7, peaceful=0.6
Comfortable and warm. Slightly more groove than chill, still low effort to listen to.

## peaceful
type: mood
energy: 0.15-0.35
synonyms: peaceful, serene, tranquil, calm, quiet, soothing, gentle
related: chill=0.5, relaxed=0.6
Very low tension and usually acoustic or ambient. Suits sleep, reading, and meditation.

## intense
type: mood
energy: 0.85-0.97
synonyms: intense, hype, hyped, pumped, pump up, aggressive, heavy, adrenaline
related: angry=0.4, euphoric=0.5, confident=0.4
Fast, loud, and driving. Built for workouts and sprints rather than background listening.

## angry
type: mood
energy: 0.90-0.98
synonyms: angry, rage, furious, mad, vent, venting
related: intense=0.6
Dark and aggressive. Distorted guitars and fast tempos.

## focused
type: mood
energy: 0.30-0.45
synonyms: focused, focus, concentrate, concentration, productive, deep work, flow state
related: chill=0.4, peaceful=0.3
Steady, repetitive, and usually without vocals so it does not compete with reading or writing.

## confident
type: mood
energy: 0.70-0.85
synonyms: confident, swagger, bold, boss, strut
related: intense=0.4, happy=0.3
Mid-tempo with a strong beat. Feels like walking into a room like you own it.

## romantic
type: mood
energy: 0.35-0.55
synonyms: romantic, love, date night, sensual, in love
related: relaxed=0.4
Warm, smooth, and intimate. R&B and soft classical often fit.

## nostalgic
type: mood
energy: 0.35-0.60
synonyms: nostalgic, throwback, memories, reminiscing, old times
related: melancholy=0.3, peaceful=0.3
Songs that look backward. Often warm and acoustic.

## melancholy
type: mood
energy: 0.20-0.40
synonyms: melancholy, melancholic, wistful, bittersweet, somber, sombre
related: moody=0.5
Low valence and slow. Reflective rather than angry.

## sad
type: mood
energy: 0.20-0.40
synonyms: sad, sadness, heartbroken, heartbreak, breakup, blue, crying, feeling down, gloomy
maps_to: melancholy
related: moody=0.5
"Sad" is not a catalog mood. The closest catalog mood is melancholy, with moody as a weaker neighbour.

## euphoric
type: mood
energy: 0.85-0.97
synonyms: euphoric, ecstatic, rave, festival, peak time
related: happy=0.5, intense=0.5
Big drops and high energy with high valence. The top of a party arc.

## uplifting
type: mood
energy: 0.55-0.75
synonyms: uplifting, hopeful, inspiring, motivational, optimistic
related: happy=0.6
Positive and forward moving without being frantic.

## moody
type: mood
energy: 0.45-0.70
synonyms: moody, brooding, atmospheric, broody, dark
related: melancholy=0.5
Minor-key and nocturnal. Tension without aggression.
