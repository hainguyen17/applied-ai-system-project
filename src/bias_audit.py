"""Bias audit: measures the weaknesses the Module 3 model card described, before and after the extension.

    python -m src.bias_audit            # plain table
    python -m src.bias_audit --markdown

Metrics over a fixed set of listener profiles (top 5 per profile):
  genre share     share of slots filled by the listener's favorite genre (high = genre dominates)
  energy error    mean |song energy - target energy| (lower = respects the energy request)
  conflict error  energy error on the 'lofi but energy 0.9' profile, the Module 3 worst case
  popularity      mean popularity of recommended songs minus the catalog mean (positive = popularity bias)
  coverage        unique songs recommended across all profiles / catalog size
  zero-credit     genre/mood words in the 3 'word:' profiles (6 words) that earned no points in the top 5
"""
import argparse
from typing import Dict, List

from src.agent import CATALOG_FILES
from src.recommender import load_catalog, load_songs, recommend_songs
from src.retriever import Retriever


def P(genre, mood, energy, acoustic):
    return {"favorite_genre": genre, "favorite_mood": mood, "target_energy": energy, "likes_acoustic": acoustic}


PROFILES: Dict[str, Dict] = {
    "pop/happy": P("pop", "happy", 0.8, False),
    "lofi/chill": P("lofi", "chill", 0.4, True),
    "rock/intense": P("rock", "intense", 0.9, False),
    "jazz/relaxed": P("jazz", "relaxed", 0.35, True),
    "metal/angry": P("metal", "angry", 0.95, False),
    "folk/peaceful": P("folk", "peaceful", 0.3, True),
    "edm/euphoric": P("edm", "euphoric", 0.9, False),
    "r&b/romantic": P("r&b", "romantic", 0.5, False),
    "conflict: lofi but 0.9": P("lofi", "chill", 0.9, False),
    "dead-center pop": P("pop", "chill", 0.5, False),
    "word: sad": P("lofi", "sad", 0.3, True),
    "word: bossa nova": P("bossa nova", "relaxed", 0.3, True),
    "word: indie": P("indie", "happy", 0.7, False),
}


def expand(profile: Dict, retriever: Retriever, genres: set, moods: set) -> Dict:
    """What the agent's ground step does: map non-catalog words, attach related labels."""
    p = dict(profile)
    for key, kind, allowed, rel_key in (("favorite_genre", "genre", genres, "related_genres"),
                                        ("favorite_mood", "mood", moods, "related_moods")):
        label = p[key]
        chunk = next((c for c in retriever.chunks if c.kind == kind and (c.title == label or label in c.synonyms)), None)
        if chunk is None:
            continue
        if label not in allowed:
            p[key] = chunk.meta.get("maps_to") or chunk.title
        p[rel_key] = {k: v for k, v in chunk.related.items() if k != p[key]}
    return p


def audit(songs: List[Dict], mode: str, use_rag: bool, retriever: Retriever) -> Dict[str, float]:
    genres = {s["genre"] for s in songs}
    moods = {s["mood"] for s in songs}
    cat_pop = sum(s["popularity"] for s in songs) / len(songs)
    share = err = pop = 0.0
    seen = set()
    zero = 0
    conflict = 0.0
    for name, prof in PROFILES.items():
        p = expand(prof, retriever, genres, moods) if use_rag else prof
        recs = recommend_songs(p, songs, k=5, mode=mode)
        share += sum(s["genre"] == p["favorite_genre"] for s, _, _ in recs) / 5
        e = sum(abs(s["energy"] - p["target_energy"]) for s, _, _ in recs) / 5
        err += e
        pop += sum(s["popularity"] for s, _, _ in recs) / 5 - cat_pop
        seen.update(s["id"] for s, _, _ in recs)
        if name.startswith("conflict"):
            conflict = e
        if name.startswith("word:"):
            reasons = " ".join(r for _, _, r in recs)
            zero += "genre match" not in reasons and "related genre" not in reasons
            zero += "mood match" not in reasons and "related mood" not in reasons
    n = len(PROFILES)
    return {"genre share": share / n, "energy error": err / n, "conflict error": conflict,
            "popularity bias": pop / n, "coverage": len(seen) / len(songs), "zero-credit words": zero}


CONFIGS = [
    ("Module 3 (18 songs, balanced)", "orig", "balanced", False),
    ("60 songs, balanced", "full", "balanced", False),
    ("60 songs, balanced + RAG labels", "full", "balanced", True),
    ("60 songs, energy-focused + RAG labels", "full", "energy-focused", True),
]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args(argv)
    retriever = Retriever.from_dir()
    data = {"orig": load_songs(CATALOG_FILES[0]), "full": load_catalog(CATALOG_FILES)}
    rows = [(label, audit(data[cat], mode, rag, retriever)) for label, cat, mode, rag in CONFIGS]
    keys = list(rows[0][1])
    if args.markdown:
        print("| Configuration | " + " | ".join(keys) + " |")
        print("|---" * (len(keys) + 1) + "|")
        for label, m in rows:
            print(f"| {label} | " + " | ".join(f"{m[k]:.2f}" if isinstance(m[k], float) else str(m[k]) for k in keys) + " |")
    else:
        print(f"{'configuration':<40}" + "".join(f"{k:>18}" for k in keys))
        for label, m in rows:
            print(f"{label:<40}" + "".join(f"{m[k]:>18.2f}" if isinstance(m[k], float) else f"{m[k]:>18}" for k in keys))


if __name__ == "__main__":
    main()
