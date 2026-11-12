"""Guardrails: everything that sits between untrusted input (the user, the LLM) and the scorer.

- check_request: rejects empty, oversized, prompt-injection, and clearly off-topic requests.
- validate_profile: turns a parser's raw dict into a safe profile. Clamps numbers, drops unknown
  keys and labels, and records a warning for every change so nothing is silently rewritten.
- check_intro: makes sure LLM-written prose only names songs that are actually in the playlist.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

MAX_REQUEST_CHARS = 500
MIN_MINUTES, MAX_MINUTES = 5, 120
MIN_SONGS, MAX_SONGS = 3, 20
ARCS = ("flat", "build", "peak", "wind-down")

_INJECTION = re.compile(
    r"ignore (all |any |the )?(previous|prior|above) (instructions|prompts?)|system prompt|"
    r"you are now|developer mode|jailbreak|disregard (your|the) rules",
    re.I,
)
_OFF_TOPIC = re.compile(
    r"\b(write|fix|debug) (me )?(a |some |my )?(python|code|script|essay|program)\b|"
    r"\b(weather|stock price|bank account|password|medical advice|diagnose)\b",
    re.I,
)
MUSIC_WORDS = re.compile(r"\b(music|songs?|playlist|tracks?|listen|listening|beats|tunes|vibes?|mix)\b", re.I)


@dataclass
class RequestCheck:
    ok: bool
    text: str
    status: str = "ok"          # ok | refused | error
    message: str = ""


def check_request(text: Optional[str]) -> RequestCheck:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return RequestCheck(False, "", "error", "Describe what you want to listen to, e.g. 'chill songs for studying'.")
    if len(cleaned) > MAX_REQUEST_CHARS:
        return RequestCheck(False, cleaned[:MAX_REQUEST_CHARS], "error",
                            f"Keep the request under {MAX_REQUEST_CHARS} characters.")
    if _INJECTION.search(cleaned):
        return RequestCheck(False, cleaned, "refused",
                            "That looks like an instruction to change how I work. I only build playlists from the catalog.")
    if _OFF_TOPIC.search(cleaned) and not MUSIC_WORDS.search(cleaned):
        return RequestCheck(False, cleaned, "refused",
                            "I can only help with playlists. Try describing a mood, genre, or activity.")
    return RequestCheck(True, cleaned)


@dataclass
class Vocabulary:
    genres: Set[str]
    moods: Set[str]
    tags: Set[str]

    @classmethod
    def from_catalog(cls, songs: List[Dict]) -> "Vocabulary":
        return cls(
            {s["genre"].lower() for s in songs},
            {s["mood"].lower() for s in songs},
            {t.lower() for s in songs for t in s.get("mood_tags", [])},
        )


@dataclass
class ValidatedRequest:
    profile: Dict                       # keys understood by recommender.score_song
    constraints: Dict                   # playlist-level: arc, duration_min, num_songs, energy_range
    warnings: List[str] = field(default_factory=list)


PROFILE_KEYS = {"favorite_genre", "favorite_mood", "target_energy", "likes_acoustic",
                "preferred_tags", "prefers_instrumental"}
CONSTRAINT_KEYS = {"arc", "duration_min", "num_songs", "energy_range"}
META_KEYS = {"activity", "confidence", "sources"}


def _as_bool(value) -> Optional[bool]:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str) and value.lower() in ("true", "yes", "1"):
        return True
    if isinstance(value, str) and value.lower() in ("false", "no", "0"):
        return False
    return None


def _clamp(value, lo, hi, name: str, warnings: List[str], cast=float):
    try:
        v = cast(value)
    except (TypeError, ValueError):
        warnings.append(f"Ignored {name}={value!r}: not a number.")
        return None
    if v < lo or v > hi:
        clamped = min(max(v, lo), hi)
        warnings.append(f"{name} {v} is outside {lo}-{hi}; using {clamped}.")
        return clamped
    return v


def validate_profile(raw: Dict, vocab: Vocabulary) -> ValidatedRequest:
    warnings: List[str] = []
    profile: Dict = {}
    constraints: Dict = {}

    for key in sorted(set(raw) - PROFILE_KEYS - CONSTRAINT_KEYS - META_KEYS):
        warnings.append(f"Dropped unknown field '{key}' from the parser output.")

    for key, allowed in (("favorite_genre", vocab.genres), ("favorite_mood", vocab.moods)):
        value = raw.get(key)
        if value in (None, ""):
            continue
        label = str(value).strip().lower()
        if label in allowed:
            profile[key] = label
        else:
            warnings.append(f"'{label}' is not a {key.split('_')[1]} in the catalog; ignoring it.")

    if raw.get("target_energy") is not None:
        energy = _clamp(raw["target_energy"], 0.0, 1.0, "target_energy", warnings)
        if energy is not None:
            profile["target_energy"] = round(energy, 2)

    acoustic = _as_bool(raw.get("likes_acoustic"))
    if acoustic is not None:
        profile["likes_acoustic"] = acoustic

    instrumental = _as_bool(raw.get("prefers_instrumental"))
    if instrumental is not None:
        profile["prefers_instrumental"] = instrumental

    tags = [str(t).strip().lower() for t in raw.get("preferred_tags") or []]
    unknown = [t for t in tags if t not in vocab.tags]
    if unknown:
        warnings.append(f"Ignored unknown mood tags: {', '.join(unknown)}.")
    known = [t for t in dict.fromkeys(tags) if t in vocab.tags]
    if known:
        profile["preferred_tags"] = known

    arc = raw.get("arc")
    if arc:
        if str(arc).lower() in ARCS:
            constraints["arc"] = str(arc).lower()
        else:
            warnings.append(f"Unknown arc '{arc}'; using a flat playlist.")

    if raw.get("duration_min") is not None:
        minutes = _clamp(raw["duration_min"], MIN_MINUTES, MAX_MINUTES, "duration_min", warnings, int)
        if minutes is not None:
            constraints["duration_min"] = minutes

    if raw.get("num_songs") is not None:
        n = _clamp(raw["num_songs"], MIN_SONGS, MAX_SONGS, "num_songs", warnings, int)
        if n is not None:
            constraints["num_songs"] = n

    rng = raw.get("energy_range")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        lo = _clamp(rng[0], 0.0, 1.0, "energy_range low", warnings)
        hi = _clamp(rng[1], 0.0, 1.0, "energy_range high", warnings)
        if lo is not None and hi is not None:
            constraints["energy_range"] = (min(lo, hi), max(lo, hi))

    return ValidatedRequest(profile, constraints, warnings)


_QUOTED = re.compile(r"[\"“]([^\"”]{3,60})[\"”]")


def check_intro(text: str, playlist_titles: List[str], catalog_titles: List[str],
                request: str = "") -> Tuple[bool, str]:
    """An intro may only name songs that are in this playlist. Returns (ok, reason).
    Quoting the listener's own request back is allowed."""
    if not text or len(text) > 600:
        return False, "intro empty or too long"
    in_playlist = {t.lower() for t in playlist_titles}
    lowered = text.lower()
    for title in catalog_titles:
        if title.lower() not in in_playlist and re.search(r"\b" + re.escape(title.lower()) + r"\b", lowered):
            return False, f"intro mentions '{title}', which is not in the playlist"
    catalog = {t.lower() for t in catalog_titles}
    for quoted in _QUOTED.findall(text):
        q = quoted.lower().strip()
        if q not in in_playlist and q not in catalog and q not in request.lower():
            return False, f"intro quotes '{quoted}', which is not a catalog song"
    return True, ""
