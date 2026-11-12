"""Human-in-the-loop feedback: a thumbs up or down on one song nudges the scoring weights.

The change is small (STEP points per matched feature), bounded (MIN_W..MAX_W), and always
reported back to the listener in plain words, so the system never retunes itself silently.
A disliked song is also excluded from the next run.
"""
from dataclasses import replace
from typing import Dict, List, Tuple

from src.recommender import Weights

STEP = 0.25
MIN_W, MAX_W = 0.0, 4.0
ENERGY_CLOSE = 0.15


def matched_features(profile: Dict, song: Dict) -> List[str]:
    """Which scoring features this song earned points on for this profile."""
    feats = []
    genre = song["genre"].lower()
    if genre == profile.get("favorite_genre") or genre in profile.get("related_genres", {}):
        feats.append("genre")
    mood = song["mood"].lower()
    if mood == profile.get("favorite_mood") or mood in profile.get("related_moods", {}):
        feats.append("mood")
    if profile.get("target_energy") is not None and abs(song["energy"] - profile["target_energy"]) <= ENERGY_CLOSE:
        feats.append("energy")
    if set(profile.get("preferred_tags") or []) & set(song.get("mood_tags") or []):
        feats.append("tags")
    return feats


def apply_feedback(weights: Weights, profile: Dict, song: Dict, liked: bool) -> Tuple[Weights, str]:
    feats = matched_features(profile, song)
    if not feats:
        verb = "liked" if liked else "disliked"
        return weights, f"You {verb} '{song['title']}', but it did not match any weighted feature, so weights are unchanged."
    delta = STEP if liked else -STEP
    changes: Dict[str, float] = {}
    notes = []
    for f in feats:
        old = getattr(weights, f)
        new = round(min(MAX_W, max(MIN_W, old + delta)), 2)
        if new != old:
            changes[f] = new
            notes.append(f"{f} {old:.2f} -> {new:.2f}")
    verb = "liked" if liked else "disliked"
    summary = (f"You {verb} '{song['title']}' (matched {', '.join(feats)}). "
               + ("; ".join(notes) if notes else "Weights already at their limit.") + ".")
    if not liked:
        summary += " It will be left out of the next playlist."
    return replace(weights, **changes), summary
