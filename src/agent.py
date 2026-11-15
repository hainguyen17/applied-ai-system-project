"""VibeMatch Agent: turns a free-text request into a verified, explained playlist.

Pipeline (each step is recorded in the trace):

  1. guard      reject empty, oversized, injection, and off-topic requests
  2. retrieve   search knowledge/ for moods, genres, and activities the request mentions (RAG)
  3. parse      LLM (or offline parser) turns request + retrieved chunks into a raw profile
  4. ground     validate the raw profile; map non-catalog labels with the glossary ('sad' -> melancholy)
                and attach related labels for partial credit
  5. clarify    if nothing usable was found, stop and ask the listener a question (human in the loop)
  6. score      the original Module 3 score_song() ranks every catalog song
  7. draft      take the top songs and lay them out along the requested energy arc
  8. verify     run playlist checks (duration, instrumental, tempo jumps, artist spacing, arc fit...)
  9. repair     swap the songs that broke a check, then verify again (max MAX_ROUNDS)
 10. confidence combine check pass rate, match strength, and parser confidence
 11. explain    per-song reasons with citations, plus an intro that is checked for invented songs
"""
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from src.guardrails import ValidatedRequest, Vocabulary, check_intro, check_request, validate_profile
from src.llm import LLMError, RuleBasedParser, get_llm
from src.recommender import MODES, Weights, get_weights, load_catalog, score_song
from src.retriever import Chunk, Retriever

ROOT = os.path.dirname(os.path.dirname(__file__))
CATALOG_FILES = [os.path.join(ROOT, "data", "songs.csv"), os.path.join(ROOT, "data", "songs_expansion.csv")]
TRACE_DIR = os.path.join(ROOT, "logs", "traces")

MAX_ROUNDS = 3
MAX_TEMPO_JUMP = 30          # bpm between consecutive songs
ARC_TOLERANCE = 0.15         # mean |energy - slot target| allowed
DURATION_TOLERANCE = 0.12    # +/- share of the requested length
ARC_WEIGHT = 3.0             # how strongly slot fit counts when picking a replacement
DEFAULT_SONGS = 5
REVIEW_THRESHOLD = 0.6
GENRE_COVERAGE = 0.4         # share of songs that must be the requested genre

log = logging.getLogger("vibematch.agent")


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    positions: List[int] = field(default_factory=list)


@dataclass
class Item:
    song: Dict
    score: float
    slot: int
    segment: str
    target_energy: float
    reasons: List[str]
    citations: List[str]


@dataclass
class Result:
    status: str                         # ok | needs_clarification | refused | error
    request: str
    message: str = ""
    profile: Dict = field(default_factory=dict)
    constraints: Dict = field(default_factory=dict)
    items: List[Item] = field(default_factory=list)
    checks: List[Check] = field(default_factory=list)
    confidence: float = 0.0
    needs_review: bool = False
    intro: str = ""
    warnings: List[str] = field(default_factory=list)
    rounds: int = 0
    llm: str = ""
    trace: List[Dict] = field(default_factory=list)
    trace_path: str = ""

    @property
    def total_minutes(self) -> float:
        return round(sum(i.song.get("duration_sec", 0) for i in self.items) / 60, 1)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["total_minutes"] = self.total_minutes
        return d


def slot_targets(n: int, arc: str, target: float, rng: Optional[Tuple[float, float]]) -> List[Tuple[float, str]]:
    lo, hi = rng if rng else (max(0.0, target - 0.15), min(1.0, target + 0.15))
    out = []
    for i in range(n):
        x = i / (n - 1) if n > 1 else 0.0
        if arc == "build":
            t, seg = lo + (hi - lo) * x, ("opening" if x < 0.34 else "build" if x < 0.8 else "top")
        elif arc == "wind-down":
            t, seg = hi - (hi - lo) * x, ("settle" if x < 0.5 else "fade")
        elif arc == "peak":
            t = lo + (hi - lo) * min(1.0, 2.5 * min(x, 1 - x))
            seg = "warm-up" if x < 0.25 else "cool-down" if x > 0.75 else "peak"
        else:
            t, seg = target, "steady"
        out.append((round(t, 2), seg))
    return out


class VibeMatchAgent:
    def __init__(
        self,
        catalog: Optional[List[Dict]] = None,
        retriever: Optional[Retriever] = None,
        llm=None,
        mode: Union[str, Weights] = "balanced",
        use_rag: bool = True,
        few_shot: bool = True,
        exclude_ids: Optional[List[int]] = None,
        save_trace: bool = False,
    ):
        self.catalog = catalog if catalog is not None else load_catalog(CATALOG_FILES)
        self.by_id = {s["id"]: s for s in self.catalog}
        self.vocab = Vocabulary.from_catalog(self.catalog)
        self.retriever = retriever or Retriever.from_dir()
        self.llm = llm or get_llm(self.vocab)
        self.fallback = RuleBasedParser(self.vocab)
        self.weights = get_weights(mode)
        self.use_rag = use_rag
        self.few_shot = few_shot
        self.exclude_ids = set(exclude_ids or [])
        self.save_trace = save_trace
        self.avg_duration = sum(s.get("duration_sec", 210) for s in self.catalog) / len(self.catalog)

    # ---- trace helper -------------------------------------------------------------------
    def _step(self, result: Result, name: str, started: float, **data) -> None:
        entry = {"step": name, "ms": round((time.perf_counter() - started) * 1000, 1), **data}
        result.trace.append(entry)
        log.info("%s %s", name, json.dumps(data, default=str)[:300])

    # ---- grounding ----------------------------------------------------------------------
    def _glossary(self, label: str, kind: str) -> Optional[Chunk]:
        for c in self.retriever.chunks:
            if c.kind == kind and (c.title == label or label in c.synonyms):
                return c
        return None

    def _ground(self, raw: Dict, result: Result) -> ValidatedRequest:
        raw = dict(raw)
        citations: Dict[str, str] = {}
        related: Dict[str, Dict[str, float]] = {}
        for key, kind, allowed in (("favorite_genre", "genre", self.vocab.genres),
                                   ("favorite_mood", "mood", self.vocab.moods)):
            label = str(raw.get(key) or "").strip().lower()
            if not label:
                continue
            chunk = self._glossary(label, kind) if self.use_rag else None
            if label not in allowed and chunk is not None:
                target = chunk.meta.get("maps_to") or chunk.title
                if target in allowed:
                    result.warnings.append(f"'{label}' is not a catalog {kind}; using '{target}' ({chunk.cite}).")
                    raw[key] = target
                    citations[key] = chunk.cite
            if chunk is not None:
                related[kind] = {k: v for k, v in chunk.related.items() if k != raw.get(key)}
                citations.setdefault(key, chunk.cite)
                rng = chunk.meta.get("energy")
                if kind == "mood" and rng and raw.get("target_energy") is None:
                    lo, hi = (float(x) for x in rng.split("-"))
                    raw["target_energy"] = round((lo + hi) / 2, 2)
                    raw.setdefault("energy_range", [lo, hi])
                    result.warnings.append(f"No energy level given; using the typical range for "
                                           f"'{chunk.title}' ({rng}, {chunk.cite}).")
        validated = validate_profile(raw, self.vocab)
        if related.get("genre") and "favorite_genre" in validated.profile:
            validated.profile["related_genres"] = related["genre"]
        if related.get("mood") and "favorite_mood" in validated.profile:
            validated.profile["related_moods"] = related["mood"]
        validated.profile.setdefault("target_energy", 0.5)
        validated.profile.setdefault("likes_acoustic", validated.profile["target_energy"] < 0.5)
        validated.constraints["citations"] = citations
        return validated

    # ---- playlist building --------------------------------------------------------------
    def _fits_hard(self, song: Dict, profile: Dict) -> bool:
        if song["id"] in self.exclude_ids:
            return False
        want = profile.get("prefers_instrumental")
        return want is None or bool(song.get("is_instrumental")) == bool(want)

    def _num_slots(self, constraints: Dict) -> int:
        if constraints.get("duration_min"):
            return max(3, min(20, round(constraints["duration_min"] * 60 / self.avg_duration)))
        return constraints.get("num_songs", DEFAULT_SONGS)

    def _draft(self, scored: List[Tuple[Dict, float, List[str]]], targets) -> List[Tuple[Dict, float, List[str]]]:
        """Naive first draft: the top-n songs by score, ordered to follow the arc. No constraints yet."""
        top = sorted(scored, key=lambda x: (-x[1], x[0]["id"]))[: len(targets)]
        by_energy = sorted(top, key=lambda x: x[0]["energy"])
        slot_order = sorted(range(len(targets)), key=lambda i: (targets[i][0], i))
        placed: List = [None] * len(targets)
        for slot, entry in zip(slot_order, by_energy):
            placed[slot] = entry
        return placed

    def _verify(self, playlist, targets, profile: Dict, constraints: Dict) -> List[Check]:
        songs = [p[0] for p in playlist]
        checks = []
        ghosts = [i for i, s in enumerate(songs) if s["id"] not in self.by_id]
        checks.append(Check("grounded_in_catalog", not ghosts, "every song exists in data/", ghosts))
        seen, dups = set(), []
        for i, s in enumerate(songs):
            if s["id"] in seen:
                dups.append(i)
            seen.add(s["id"])
        checks.append(Check("no_duplicates", not dups, f"{len(dups)} repeated", dups))
        excluded = [i for i, s in enumerate(songs) if s["id"] in self.exclude_ids]
        checks.append(Check("respects_feedback", not excluded, f"{len(excluded)} disliked songs", excluded))
        if profile.get("prefers_instrumental") is not None:
            want = profile["prefers_instrumental"]
            bad = [i for i, s in enumerate(songs) if bool(s.get("is_instrumental")) != want]
            label = "instrumental only" if want else "vocals only"
            checks.append(Check("instrumental", not bad, f"{label}; {len(bad)} violate", bad))
        jumps = [i for i in range(1, len(songs)) if abs(songs[i]["tempo_bpm"] - songs[i - 1]["tempo_bpm"]) > MAX_TEMPO_JUMP]
        checks.append(Check("tempo_transitions", not jumps, f"max jump {MAX_TEMPO_JUMP} bpm; {len(jumps)} too big", jumps))
        repeats = [i for i in range(1, len(songs)) if songs[i]["artist"] == songs[i - 1]["artist"]]
        checks.append(Check("artist_spacing", not repeats, f"{len(repeats)} back-to-back artists", repeats))
        errors = [abs(s["energy"] - t) for s, (t, _) in zip(songs, targets)]
        mean_err = sum(errors) / len(errors)
        off = [i for i, e in enumerate(errors) if e > ARC_TOLERANCE * 1.5]
        checks.append(Check("arc_fit", mean_err <= ARC_TOLERANCE,
                            f"mean energy error {mean_err:.2f} (limit {ARC_TOLERANCE})", off if mean_err > ARC_TOLERANCE else []))
        if constraints.get("duration_min"):
            total = sum(s.get("duration_sec", 0) for s in songs) / 60
            want = constraints["duration_min"]
            ok = abs(total - want) <= want * DURATION_TOLERANCE
            checks.append(Check("duration", ok, f"{total:.1f} min vs {want} requested"))
        genre = profile.get("favorite_genre")
        if genre:
            # Not repairable by swapping: if it fails, the request itself conflicts with the catalog.
            n_match = sum(s["genre"] == genre for s in songs)
            ok = n_match / len(songs) >= GENRE_COVERAGE
            checks.append(Check("honors_genre", ok, f"{n_match}/{len(songs)} songs are {genre}"))
        return checks

    def _pick(self, scored, playlist, pos: int, targets, profile: Dict, strict: bool):
        used = {p[0]["id"] for i, p in enumerate(playlist) if i != pos}
        prev = playlist[pos - 1][0] if pos > 0 else None
        nxt = playlist[pos + 1][0] if pos + 1 < len(playlist) else None
        target = targets[pos][0]
        best, best_val = None, -math.inf
        for entry in scored:
            s = entry[0]
            if s["id"] in used or not self._fits_hard(s, profile):
                continue
            if strict:
                if any(n and (abs(s["tempo_bpm"] - n["tempo_bpm"]) > MAX_TEMPO_JUMP or s["artist"] == n["artist"])
                       for n in (prev, nxt)):
                    continue
            val = entry[1] - ARC_WEIGHT * abs(s["energy"] - target)
            if val > best_val or (val == best_val and best and s["id"] < best[0]["id"]):
                best, best_val = entry, val
        return best

    def _repair(self, scored, playlist, targets, checks: List[Check], profile: Dict, constraints: Dict):
        swaps = []
        failing = [c for c in checks if not c.passed]
        duration = next((c for c in failing if c.name == "duration"), None)
        if duration:
            total = sum(p[0].get("duration_sec", 0) for p in playlist) / 60
            n = len(playlist) + (1 if total < constraints["duration_min"] else -1)
            n = max(3, min(20, n))
            new_targets = slot_targets(n, constraints.get("arc", "flat"), profile["target_energy"], constraints.get("energy_range"))
            kept = sorted(playlist, key=lambda p: -p[1])[:n]
            playlist = self._draft(kept + scored, new_targets) if n > len(playlist) else self._draft(kept, new_targets)
            swaps.append(f"resized playlist to {n} songs for duration")
            return playlist, new_targets, swaps
        positions = sorted({p for c in failing for p in c.positions})
        for pos in positions:
            old = playlist[pos][0]
            new = self._pick(scored, playlist, pos, targets, profile, strict=True) or \
                self._pick(scored, playlist, pos, targets, profile, strict=False)
            if new and new[0]["id"] != old["id"]:
                playlist[pos] = new
                swaps.append(f"slot {pos + 1}: {old['title']} -> {new[0]['title']}")
        return playlist, targets, swaps

    def _max_points(self, profile: Dict) -> float:
        w = self.weights
        pts = w.energy + w.acoustic
        pts += w.genre if profile.get("favorite_genre") else 0
        pts += w.mood if profile.get("favorite_mood") else 0
        pts += w.tags if profile.get("preferred_tags") else 0
        pts += w.instrumental if profile.get("prefers_instrumental") is not None else 0
        return pts or 1.0

    # ---- main entry ---------------------------------------------------------------------
    def run(self, request: str) -> Result:
        result = Result(status="ok", request=request or "", llm=getattr(self.llm, "name", "unknown"))
        t = time.perf_counter()
        guard = check_request(request)
        self._step(result, "guard", t, ok=guard.ok, status=guard.status)
        if not guard.ok:
            result.status, result.message = guard.status, guard.message
            return self._finish(result)
        text = guard.text

        t = time.perf_counter()
        chunks = self.retriever.search(text, k=8) if self.use_rag else []
        self._step(result, "retrieve", t, enabled=self.use_rag,
                   hits=[{"cite": c.cite, "score": c.score, "matched": c.matched} for c in chunks])

        t = time.perf_counter()
        try:
            raw = self.llm.parse(text, chunks, few_shot=self.few_shot)
            parser = result.llm
        except LLMError as exc:
            log.warning("LLM parse failed (%s); falling back to the offline parser", exc)
            result.warnings.append(f"{result.llm} failed ({exc}); used the offline parser instead.")
            raw = self.fallback.parse(text, chunks)
            parser = self.fallback.name
        self._step(result, "parse", t, parser=parser, raw=raw)

        t = time.perf_counter()
        validated = self._ground(raw, result)
        result.warnings += validated.warnings
        result.profile, result.constraints = validated.profile, validated.constraints
        self._step(result, "ground", t, profile=result.profile, constraints=result.constraints,
                   warnings=validated.warnings)

        signals = [k for k in ("favorite_genre", "favorite_mood") if k in result.profile] + \
            (["activity"] if raw.get("activity") else []) + \
            (["energy"] if raw.get("target_energy") is not None else [])
        if not signals:
            result.status = "needs_clarification"
            result.message = ("I could not find a mood, genre, activity, or energy level in that request. "
                              "What is the music for? For example: 'chill lofi for studying' or "
                              "'high-energy workout mix, 30 minutes'.")
            self._step(result, "clarify", time.perf_counter(), signals=signals)
            return self._finish(result)

        t = time.perf_counter()
        profile, constraints = result.profile, result.constraints
        scored = []
        for song in self.catalog:
            s, reasons = score_song(profile, song, self.weights)
            scored.append((song, s, reasons))
        scored.sort(key=lambda x: (-x[1], x[0]["id"]))
        self._step(result, "score", t, top=[(x[0]["title"], round(x[1], 2)) for x in scored[:8]])

        t = time.perf_counter()
        n = self._num_slots(constraints)
        arc = constraints.get("arc", "flat")
        targets = slot_targets(n, arc, profile["target_energy"], constraints.get("energy_range"))
        playlist = self._draft(scored, targets)
        self._step(result, "draft", t, slots=n, arc=arc, songs=[p[0]["title"] for p in playlist])

        checks: List[Check] = []
        for round_no in range(MAX_ROUNDS + 1):
            t = time.perf_counter()
            checks = self._verify(playlist, targets, profile, constraints)
            failed = [c.name for c in checks if not c.passed]
            self._step(result, "verify", t, round=round_no, failed=failed,
                       checks={c.name: c.detail for c in checks})
            if not failed or round_no == MAX_ROUNDS:
                break
            t = time.perf_counter()
            playlist, targets, swaps = self._repair(scored, playlist, targets, checks, profile, constraints)
            result.rounds = round_no + 1
            self._step(result, "repair", t, round=round_no + 1, swaps=swaps)
            if not swaps:
                break
        result.checks = checks
        conflict = next((c for c in checks if c.name == "honors_genre" and not c.passed), None)
        if conflict:
            result.warnings.append(
                f"Only {conflict.detail}: the catalog has too few {profile['favorite_genre']} songs that fit the "
                f"other parts of the request. Loosen the energy or the genre and try again.")

        t = time.perf_counter()
        pass_rate = sum(c.passed for c in checks) / len(checks)
        match = min(1.0, (sum(p[1] for p in playlist) / len(playlist)) / self._max_points(profile))
        parse_conf = float(raw.get("confidence", 0.5) or 0.5)
        result.confidence = round(0.5 * pass_rate + 0.3 * match + 0.2 * parse_conf, 2)
        result.needs_review = result.confidence < REVIEW_THRESHOLD or pass_rate < 1.0
        self._step(result, "confidence", t, pass_rate=round(pass_rate, 2), match=round(match, 2),
                   parse=parse_conf, confidence=result.confidence)

        t = time.perf_counter()
        cites = constraints.get("citations", {})
        for i, ((song, s, reasons), (target, seg)) in enumerate(zip(playlist, targets)):
            item_cites = []
            if song["genre"] == profile.get("favorite_genre") and "favorite_genre" in cites:
                item_cites.append(cites["favorite_genre"])
            if song["genre"] in profile.get("related_genres", {}):
                item_cites.append(cites.get("favorite_genre", "knowledge/genres.md"))
            if song["mood"] == profile.get("favorite_mood") and "favorite_mood" in cites:
                item_cites.append(cites["favorite_mood"])
            if song["mood"] in profile.get("related_moods", {}):
                item_cites.append(cites.get("favorite_mood", "knowledge/moods.md"))
            result.items.append(Item(song, round(s, 2), i + 1, seg, target,
                                     reasons + [f"{seg} slot, target energy {target:.2f}"],
                                     list(dict.fromkeys(item_cites))))
        summaries = [{k: it.song[k] for k in ("title", "artist", "genre", "energy", "duration_sec")}
                     for it in result.items]
        try:
            intro = self.llm.write_intro(text, summaries, arc)
        except LLMError as exc:
            result.warnings.append(f"Intro generation failed ({exc}); used a template.")
            intro = self.fallback.write_intro(text, summaries, arc)
        ok, why = check_intro(intro, [s["title"] for s in summaries], [s["title"] for s in self.catalog], text)
        if not ok:
            result.warnings.append(f"Rejected the generated intro: {why}. Used a template instead.")
            intro = self.fallback.write_intro(text, summaries, arc)
        result.intro = intro
        self._step(result, "explain", t, intro_ok=ok)
        return self._finish(result)

    def _finish(self, result: Result) -> Result:
        if self.save_trace:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            path = os.path.join(TRACE_DIR, f"{stamp}.json")
            try:
                os.makedirs(TRACE_DIR, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"request": result.request, "status": result.status, "llm": result.llm,
                               "confidence": result.confidence, "trace": result.trace}, f, indent=2, default=str)
                result.trace_path = path
            except OSError as e:
                # The trace is a debugging aid; failing to save it must not lose the playlist.
                log.warning("could not save trace to %s: %s", path, e)
                result.warnings.append("The trace could not be saved to disk; the playlist is unaffected.")
        return result


def configure_logging(level: int = logging.INFO) -> None:
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    handler = logging.FileHandler(os.path.join(ROOT, "logs", "vibematch.log"), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger("vibematch")
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(level)
