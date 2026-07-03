"""music21 <-> compound-event codec for the MMT-BERT representation.

An "event" is a 7-field integer row — see :mod:`ml.events` for the schema.
This module converts music21 scores to per-measure event rows and back.

The parse-once / emit-cheap split of the old flat-token tokenizer is kept:

  * :func:`collect_score` walks a score with music21 ONCE and returns a plain
    pickle-friendly ``collected`` tuple.
  * :func:`emit_measure_events` re-assembles per-measure event rows from
    ``collected`` for arbitrary staff subsets — cheap pure python, which is
    what partial-staff FIM and the token cache rely on.
  * :func:`flatten_measure_events` concatenates measure blocks while assigning
    window-relative bar indices (the compound representation's replacement for
    counting flat BAR tokens).

Legacy API-compat shim: ``api/routers/generate.py`` still builds its
``first_bar_prefix`` from the old flat-token constants ``BAR``, ``KEY_OFF``,
``TIME_OFF`` and ``TEMPO_OFF``.  Those names are kept here as sentinels (their
historical numeric values; the ranges do not overlap) and
:func:`parse_legacy_first_bar_prefix` converts such a prefix into
(key_idx, time_idx, tempo_idx) for the new event-based generators.
"""

from __future__ import annotations
import music21 as m21
from collections import defaultdict

from ml.events import (
    # fields / types
    F_TYPE, F_BAR, F_POS, F_PITCH, F_DUR, F_TRACK, F_VALUE, N_FIELDS,
    PAD_T, SOS_T, EOS_T, INSTR_T, BAR_T, KEY_T, TIME_T, TEMPO_T, CLEF_T,
    NOTE_T, GRACE_T, REST_T, CHORD_T, FILL_START_T, FILL_END_T, FILL_GEN_T,
    N_TYPES, FIELD_VOCAB_SIZES, FIELD_LOSS_MASK,
    # constants
    DURATIONS, N_DURATIONS, TIME_SIGS, CLEFS, N_CLEFS,
    MAX_REAL_PARTS, MAX_VOICES_PER_PART, MAX_PARTS,
    STEP_SIZE, MAX_POS, MAX_BARS,
    N_TEMPO, TEMPO_BUCKETS, N_INSTR, INSTR_NAMES,
    CHORD_KINDS, N_CHORD_KINDS,
    # constructors
    sos_event, eos_event, fill_start_event, fill_end_event, fill_gen_event,
    bar_event, key_event, time_event, tempo_event, clef_event, instr_event,
    note_event, grace_event, rest_event, chord_event,
    transpose_events,
)

# ---------------------------------------------------------------------------
# Legacy flat-token sentinels (API-compat shim; see module docstring)
# ---------------------------------------------------------------------------

BAR       = 3     # legacy flat BAR token id
KEY_OFF   = 175   # legacy KEY_SIG offset  (KEY_OFF + sharps+6)
TIME_OFF  = 188   # legacy TIME_SIG offset (TIME_OFF + TIME_SIGS idx)
TEMPO_OFF = 422   # legacy TEMPO offset    (TEMPO_OFF + bucket idx)


def parse_legacy_first_bar_prefix(
    prefix: list[int] | None,
) -> tuple[int | None, int | None, int | None]:
    """Legacy [BAR, KEY_OFF+k, TIME_OFF+t, TEMPO_OFF+b] -> (key_idx, time_idx, tempo_idx).

    Each element is optional; unknown token ids are ignored.  ``key_idx`` is the
    0..12 sharps+6 index, ``time_idx`` a TIME_SIGS index, ``tempo_idx`` a
    TEMPO_BUCKETS index; missing entries come back as None.
    """
    key_idx = time_idx = tempo_idx = None
    for t in prefix or []:
        if KEY_OFF <= t < KEY_OFF + 13:
            key_idx = t - KEY_OFF
        elif TIME_OFF <= t < TIME_OFF + len(TIME_SIGS):
            time_idx = t - TIME_OFF
        elif TEMPO_OFF <= t < TEMPO_OFF + N_TEMPO:
            tempo_idx = t - TEMPO_OFF
    return key_idx, time_idx, tempo_idx


# ---------------------------------------------------------------------------
# Small helpers (unchanged semantics from the flat-token tokenizer)
# ---------------------------------------------------------------------------

def _closest_dur_idx(qlen: float) -> int:
    return min(range(len(DURATIONS)), key=lambda i: abs(DURATIONS[i] - qlen))


def _offset_to_step(offset: float) -> int:
    """Quarter-length offset -> 1/24-ql step index, clamped to [0, MAX_POS)."""
    return min(MAX_POS - 1, max(0, round(offset / STEP_SIZE)))


def bpm_to_tempo_idx(bpm: float) -> int:
    """Map a BPM value to a TEMPO bucket index (0 to N_TEMPO-1)."""
    idx = 0
    for i, boundary in enumerate(TEMPO_BUCKETS):
        if bpm >= boundary:
            idx = i
    return idx


def instrument_to_idx(part) -> int:
    """Determine the INSTR category index (0-31) for a music21 Part."""

    instrs = list(part.getElementsByClass(m21.instrument.Instrument))
    cls_name = type(instrs[0]).__name__.lower() if instrs else ""
    part_name = (part.partName or "").lower()
    name = (part_name + " " + cls_name).strip()

    # Voice subtypes (check before generic "voice" / "bass")
    if "soprano" in name:                                      return 24
    if "mezzo" in name:                                        return 25
    if "contralto" in name or ("alto" in name and "contralt" in name): return 26
    if "alto" in name and "voice" in name:                    return 26
    if "tenor" in name:                                        return 27
    if "bariton" in name:                                      return 28
    if "bass" in name and ("voice" in name or "sing" in name or "gesang" in name): return 29
    if ("voice" in name or "vocal" in name or "gesang" in name
            or "sing" in name or "soprano" in name or "mezzo" in name): return 30
    if "choir" in name or "chorus" in name:                   return 31

    # Keyboard / plucked / etc.
    if "piano" in name or "fortepiano" in name or "klavier" in name: return 1
    if "harpsichord" in name or "cembalo" in name:            return 2
    if "organ" in name:                                        return 3
    if "accordion" in name:                                    return 4
    if "guitar" in name and "bass" not in name:               return 5
    if "bass guitar" in name or "electric bass" in name:      return 6

    # Strings
    if "violin" in name and "viola" not in name:              return 7
    if "viola" in name and "bass" not in name:                return 8
    if "cello" in name or "violoncello" in name:              return 9
    if "double bass" in name or "contrabass" in name or "doublebass" in name: return 10
    if "harp" in name:                                         return 11

    # Woodwinds
    if "flute" in name or "piccolo" in name:                  return 12
    if "oboe" in name or "english horn" in name or "cor anglais" in name: return 13
    if "clarinet" in name:                                     return 14
    if "bassoon" in name or "fagot" in name:                  return 15
    if "saxophone" in name or "saxophon" in name:             return 16

    # Brass
    if "trumpet" in name or "cornet" in name:                 return 17
    if "horn" in name:                                         return 18
    if "trombone" in name:                                     return 19
    if "tuba" in name:                                         return 20

    # Percussion
    if "timpani" in name or "timpano" in name:                return 22
    if "percussion" in name or "drum" in name or "snare" in name: return 21

    # Other
    if "celesta" in name or "celeste" in name:                return 23

    return 0  # other / unknown


def extract_part_clef(part: m21.stream.Part) -> int:
    for measure in part.getElementsByClass(m21.stream.Measure):
        for clef in measure.getElementsByClass(m21.clef.Clef):
            if clef.name in CLEFS:
                return CLEFS.index(clef.name)
    return 0  # default to treble clef


# ---------------------------------------------------------------------------
# Chord symbols
# ---------------------------------------------------------------------------

# music21 chordKind -> CHORD_KINDS index (with aliases for less common spellings).
_KIND_TO_IDX = {k: i for i, k in enumerate(CHORD_KINDS)}
_KIND_ALIASES = {
    "dominant-seventh": "dominant",
    "minor-major": "major-minor",
    "minor-major-seventh": "major-minor",
    "pedal": "power",
    "none": "other",
    "": "major",
}
_OTHER_KIND_IDX = _KIND_TO_IDX["other"]


def _chord_kind_idx(kind: str | None) -> int:
    kind = kind or ""
    if kind in _KIND_TO_IDX:
        return _KIND_TO_IDX[kind]
    return _KIND_TO_IDX.get(_KIND_ALIASES.get(kind, ""), _OTHER_KIND_IDX)


def chordsym_to_event(cs, pos: int = 0) -> list[int] | None:
    """harmony.ChordSymbol -> CHORD event row (bar=0 placeholder) or None."""
    try:
        root = cs.root()
        if root is None:
            return None
        root_pc = root.pitchClass
        kind_idx = _chord_kind_idx(getattr(cs, "chordKind", None))
        bass = cs.bass()
        bass_pc = bass.pitchClass if (bass is not None and bass.pitchClass != root_pc) else None
        return chord_event(0, pos, root_pc, kind_idx, bass_pc)
    except Exception:
        return None


def make_chordsym(root_pc: int, kind_idx: int, bass_pc: int | None = None):
    """Reconstruct a harmony.ChordSymbol from decoded event fields (or None)."""
    try:
        kwargs = {"root": m21.pitch.Pitch(midi=60 + root_pc).name}
        kind = CHORD_KINDS[kind_idx] if 0 <= kind_idx < N_CHORD_KINDS else "other"
        if kind != "other":
            kwargs["kind"] = kind
        if bass_pc is not None:
            kwargs["bass"] = m21.pitch.Pitch(midi=60 + bass_pc).name
        cs = m21.harmony.ChordSymbol(**kwargs)
        cs.duration = m21.duration.Duration(0)
        return cs
    except Exception:
        return None


# Sort key that puts CHORD (harmony) rows before note rows at the same onset,
# mirroring the old HARMONY_LANE = -2 ordering.
def _row_sort_key(row: list[int]) -> tuple[int, int]:
    lane = -1 if row[F_TYPE] == CHORD_T else row[F_TRACK]
    return (row[F_POS], lane)


# ---------------------------------------------------------------------------
# Encode: score -> collected -> per-measure event rows
# ---------------------------------------------------------------------------

def score_to_header_events(score) -> list[list[int]]:
    """One INSTR event per part (instrument category + clef), emitted after SOS."""
    parts = list(score.parts)[:MAX_REAL_PARTS]
    return [
        instr_event(part_idx, instrument_to_idx(p), extract_part_clef(p))
        for part_idx, p in enumerate(parts)
    ]


def collect_score(score) -> tuple:
    """Walk a score once and return raw per-measure event rows plus symbols.

    Returns ``(events_by_measure, key_vals, time_vals, tempo_vals, clef_changes)``:
      * events_by_measure: {m_num: [event_row, ...]} — NOTE/GRACE/REST/CHORD rows
        with a bar-field placeholder of 0 (assigned at flatten time).
      * key_vals:   {m_num: sharps (int, -6..+6)}
      * time_vals:  {m_num: TIME_SIGS index}
      * tempo_vals: {m_num: TEMPO_BUCKETS index}
      * clef_changes: {(part_idx, m_num): CLEFS index} — mid-score clef changes
        only (the opening clef lives in the header INSTR event).

    Everything is plain python (ints/lists/dicts/tuples) so it pickles cleanly
    into the token cache.  This is the expensive music21-walking half of
    tokenization; :func:`emit_measure_events` is the cheap re-emission half.
    """
    parts = list(score.parts)[:MAX_REAL_PARTS]
    if not parts:
        return {}, {}, {}, {}, {}

    events_by_measure: dict[int, list[list[int]]] = {}

    for part_idx, part in enumerate(parts):
        for measure in part.getElementsByClass(m21.stream.Measure):
            m_num = measure.number if measure.number is not None else 0
            if m_num not in events_by_measure:
                events_by_measure[m_num] = []

            # Chord symbols (lead-sheet harmony): collect once, from the top
            # staff, where they conventionally live.
            if part_idx == 0:
                for cs in measure.getElementsByClass(m21.harmony.ChordSymbol):
                    row = chordsym_to_event(cs, pos=_offset_to_step(float(cs.offset)))
                    if row is not None:
                        events_by_measure[m_num].append(row)

            if measure.voices:
                voice_streams = []
                found_v = 0
                has_filled_voice = any([(True if any([(False if el.isRest else True) for el in v.notesAndRests]) else False) for v in measure.voices])
                for v in measure.voices:
                    els = v.notesAndRests
                    if any([(False if el.isRest else True) for el in els]) or not has_filled_voice:
                        voice_streams.append((found_v, els))
                        found_v += 1
                        if found_v >= MAX_VOICES_PER_PART: break
            else:
                voice_streams = [(0, measure.notesAndRests)]

            for voice_idx, note_stream in voice_streams:
                lane = part_idx * MAX_VOICES_PER_PART + voice_idx
                for el in note_stream:
                    # ChordSymbol subclasses Chord; handled above on part 0.
                    if isinstance(el, m21.harmony.Harmony):
                        continue
                    pos = _offset_to_step(float(el.offset))
                    is_grace = el.duration.quarterLength == 0

                    if is_grace:
                        dtype = el.duration.type
                        written_ql = (m21.duration.Duration(dtype).quarterLength
                                      if dtype and dtype != 'zero' else 0.25)
                        dur_idx = _closest_dur_idx(written_ql if written_ql > 0 else 0.25)
                    else:
                        dur_idx = _closest_dur_idx(el.duration.quarterLength)

                    if el.isRest:
                        # Only add a rest when nothing else in this lane already
                        # occupies the same onset (matches old behaviour).
                        occupied = any(
                            r[F_POS] == pos and r[F_TRACK] == lane + 1
                            for r in events_by_measure[m_num]
                            if r[F_TYPE] in (NOTE_T, GRACE_T, REST_T)
                        )
                        if not occupied:
                            events_by_measure[m_num].append(rest_event(0, pos, dur_idx, lane))
                    elif el.isNote or el.isChord:
                        note_list = [el] if el.isNote else list(el.notes)
                        tied = (not is_grace and el.tie is not None
                                and el.tie.type in ('start', 'continue'))
                        for n in note_list:
                            if is_grace:
                                events_by_measure[m_num].append(
                                    grace_event(0, pos, n.pitch.midi, dur_idx, lane))
                            else:
                                events_by_measure[m_num].append(
                                    note_event(0, pos, n.pitch.midi, dur_idx, lane, tied))

    key_vals, time_vals, tempo_vals, clef_changes = _extract_measure_symbols(score)
    return events_by_measure, key_vals, time_vals, tempo_vals, clef_changes


def _extract_measure_symbols(score):
    """Per-measure KEY / TIME / TEMPO values (first part) and clef changes (all parts)."""
    parts = list(score.parts)[:MAX_REAL_PARTS]
    if not parts:
        return {}, {}, {}, {}

    key_vals: dict[int, int] = {}
    time_vals: dict[int, int] = {}
    tempo_vals: dict[int, int] = {}
    clef_changes: dict[tuple[int, int], int] = {}

    # Mid-score clef changes (the opening clef is carried by the header).
    for part_idx, part in enumerate(parts):
        current_clef: int | None = None
        for measure in part.getElementsByClass(m21.stream.Measure):
            m_num = measure.number if measure.number is not None else 0
            for c in measure.getElementsByClass(m21.clef.Clef):
                if c.name in CLEFS:
                    idx = CLEFS.index(c.name)
                    if current_clef is not None and idx != current_clef:
                        clef_changes[(part_idx, m_num)] = idx
                    current_clef = idx
                break

    for measure in parts[0].getElementsByClass(m21.stream.Measure):
        m_num = measure.number if measure.number is not None else 0

        for ks in measure.getElementsByClass(m21.key.KeySignature):
            sharps = int(ks.sharps)
            if -6 <= sharps <= 6:
                key_vals[m_num] = sharps
            break

        for ts in measure.getElementsByClass(m21.meter.TimeSignature):
            ratio = ts.ratioString
            if ratio in TIME_SIGS:
                time_vals[m_num] = TIME_SIGS.index(ratio)
            break

        for mm in measure.getElementsByClass(m21.tempo.MetronomeMark):
            if mm.number is not None:
                tempo_vals[m_num] = bpm_to_tempo_idx(float(mm.number))
            break

    return key_vals, time_vals, tempo_vals, clef_changes


def derive_bar_chords(score, collected, mode: str = "auto") -> tuple:
    """Add one analysis-derived CHORD event per bar when the score has none.

    Args:
        score: the parsed music21 score (needed for chordify).
        collected: output of :func:`collect_score` (returned updated).
        mode: "existing" — keep only chords already present (no analysis);
              "off" — strip all CHORD events;
              "auto" — if the score already has CHORD events (e.g. Wikifonia
              chord symbols) keep them, otherwise run music21 chordify analysis
              and add the most salient chord of each measure at pos 0.

    Chordify is slow; call this at cache-build time only.  Failures are
    swallowed — chords are optional context, scores without them still train.
    """
    events_by_measure, key_vals, time_vals, tempo_vals, clef_changes = collected

    if mode == "off":
        for m_num, rows in events_by_measure.items():
            events_by_measure[m_num] = [r for r in rows if r[F_TYPE] != CHORD_T]
        return events_by_measure, key_vals, time_vals, tempo_vals, clef_changes

    has_chords = any(
        r[F_TYPE] == CHORD_T for rows in events_by_measure.values() for r in rows
    )
    if mode == "existing" or has_chords:
        return collected

    try:
        chordified = score.chordify()
        for measure in chordified.getElementsByClass(m21.stream.Measure):
            m_num = measure.number if measure.number is not None else 0
            if m_num not in events_by_measure:
                continue
            chords = [c for c in measure.getElementsByClass(m21.chord.Chord)
                      if len(c.pitches) >= 2]
            if not chords:
                continue
            # Most salient chord = the one sounding longest in the measure.
            best = max(chords, key=lambda c: float(c.duration.quarterLength))
            row = _analyzed_chord_to_event(best)
            if row is not None:
                events_by_measure[m_num].append(row)
    except Exception:
        pass

    return events_by_measure, key_vals, time_vals, tempo_vals, clef_changes


def _analyzed_chord_to_event(c: m21.chord.Chord) -> list[int] | None:
    """Map an analyzed (chordified) music21 Chord to a CHORD event at pos 0."""
    try:
        root = c.root()
        if root is None:
            return None
        if c.isDominantSeventh():
            kind = "dominant"
        elif c.isDiminishedSeventh():
            kind = "diminished-seventh"
        elif c.seventh is not None and c.quality == "major":
            kind = "major-seventh"
        elif c.seventh is not None and c.quality == "minor":
            kind = "minor-seventh"
        elif c.quality in ("major", "minor", "diminished", "augmented"):
            kind = c.quality
        else:
            kind = "other"
        root_pc = root.pitchClass
        bass = c.bass()
        bass_pc = bass.pitchClass if (bass is not None and bass.pitchClass != root_pc) else None
        return chord_event(0, 0, root_pc, _KIND_TO_IDX[kind], bass_pc)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Emit: collected -> per-measure event blocks -> flat windows
# ---------------------------------------------------------------------------

def emit_measure_events(
    collected,
    part_filter: set[int] | None = None,
    measure_nums: list[int] | None = None,
) -> dict[int, list[list[int]]]:
    """Assemble per-measure event blocks from collected data (cheap, pure python).

    Each measure block is ``[BAR] [KEY?] [TIME?] [TEMPO?] [CLEF?...]`` followed
    by content rows sorted by (pos, lane) with CHORD rows first at each onset.
    All rows are fresh copies with the bar field left 0 — assign window-relative
    bar indices with :func:`flatten_measure_events`.

    Args:
        part_filter: if given, only content rows whose real part index is in the
            set are emitted (CHORD rows ride with part 0, like the old harmony
            lane).  BAR + symbols are still emitted for every measure, so a
            measure with no content for the filtered staves is a bare BAR block.
        measure_nums: explicit measure numbers to emit (in order); defaults to
            all measures present.  Numbers absent from the collected data still
            produce a bare BAR block.
    """
    events_by_measure, key_vals, time_vals, tempo_vals, clef_changes = collected
    if measure_nums is None:
        measure_nums = sorted(events_by_measure.keys())

    clefs_by_measure: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (part_idx, m_num), clef_idx in clef_changes.items():
        if part_filter is None or part_idx in part_filter:
            clefs_by_measure[m_num].append((part_idx, clef_idx))

    out: dict[int, list[list[int]]] = {}
    for m_num in measure_nums:
        block: list[list[int]] = [bar_event(0)]
        if m_num in key_vals:
            block.append(key_event(0, key_vals[m_num]))
        if m_num in time_vals:
            block.append(time_event(0, time_vals[m_num]))
        if m_num in tempo_vals:
            block.append(tempo_event(0, tempo_vals[m_num]))
        for part_idx, clef_idx in sorted(clefs_by_measure.get(m_num, [])):
            block.append(clef_event(0, part_idx, clef_idx))

        rows = events_by_measure.get(m_num, [])
        if part_filter is not None:
            rows = [
                r for r in rows
                if (0 if r[F_TYPE] == CHORD_T
                    else (r[F_TRACK] - 1) // MAX_VOICES_PER_PART) in part_filter
            ]
        block.extend(list(r) for r in sorted(rows, key=_row_sort_key))
        out[m_num] = block

    return out


def flatten_measure_events(
    measure_events: dict[int, list[list[int]]],
    measure_nums: list[int] | None = None,
    start_bar: int = 0,
) -> list[list[int]]:
    """Concatenate measure blocks, assigning bar index start_bar+i to block i.

    Mutates the (already-copied) rows produced by :func:`emit_measure_events`;
    do not pass rows straight out of a ``collected`` structure.
    """
    if measure_nums is None:
        measure_nums = sorted(measure_events.keys())
    flat: list[list[int]] = []
    for i, m_num in enumerate(measure_nums):
        bar = min(start_bar + i, MAX_BARS - 1)
        for row in measure_events.get(m_num, [bar_event(0)]):
            row[F_BAR] = bar
            flat.append(row)
    return flat


def score_to_events_by_measure(
    score, part_filter: set[int] | None = None,
) -> dict[int, list[list[int]]]:
    """music21 Score -> {measure number: event block} (bar fields left 0)."""
    collected = collect_score(score)
    if not collected[0]:
        return {}
    return emit_measure_events(collected, part_filter=part_filter)


def gap_measure_events(
    collected, gap_start: int, gap_end: int,
    part_filter: set[int] | None = None,
    start_bar: int = 0,
) -> list[list[int]]:
    """Flat event rows for gap measures [gap_start, gap_end] with bar indices
    start_bar.., restricted to part_filter.  Every measure in the range emits
    a BAR row even when the filtered staves are empty there, so kept/target
    gap blocks stay measure-aligned."""
    measure_nums = list(range(gap_start, gap_end + 1))
    blocks = emit_measure_events(collected, part_filter=part_filter, measure_nums=measure_nums)
    return flatten_measure_events(blocks, measure_nums, start_bar=start_bar)


def extract_prefix_suffix_events(
    collected, gap_start: int, gap_end: int, max_context_measures: int = 16,
) -> tuple[list[list[int]], list[list[int]], int]:
    """Extract flattened prefix and suffix event windows around a gap.

    Bar indices are window-relative: prefix bars run 0..P-1, the gap occupies
    bars P..P+G-1 (returned as ``gap_bar_start`` = P), and suffix bars continue
    at P+G.  Only the nearest ``max_context_measures`` measures on each side
    are included, matching the FIM training window.

    Returns (prefix_events, suffix_events, gap_bar_start).
    """
    events_by_measure = collected[0]
    all_nums = sorted(events_by_measure.keys())

    pre_nums  = [m for m in all_nums if m < gap_start][-max_context_measures:]
    post_nums = [m for m in all_nums if m > gap_end][:max_context_measures]
    gap_len = gap_end - gap_start + 1

    blocks = emit_measure_events(collected, measure_nums=pre_nums + post_nums)
    prefix = flatten_measure_events(blocks, pre_nums, start_bar=0)
    suffix = flatten_measure_events(blocks, post_nums, start_bar=len(pre_nums) + gap_len)
    return prefix, suffix, len(pre_nums)


# ---------------------------------------------------------------------------
# Decode: events -> music21 Score
# ---------------------------------------------------------------------------

_CLEF_CLASSES = [
    m21.clef.TrebleClef, m21.clef.BassClef, m21.clef.AltoClef,
    m21.clef.TenorClef, m21.clef.Treble8vbClef, m21.clef.Treble8vaClef,
    m21.clef.Bass8vbClef,
]

_INSTR_DISPLAY = [
    "Part", "Piano", "Harpsichord", "Organ", "Accordion",
    "Guitar", "Bass Guitar", "Violin", "Viola", "Cello",
    "Double Bass", "Harp", "Flute", "Oboe", "Clarinet",
    "Bassoon", "Saxophone", "Trumpet", "Horn", "Trombone",
    "Tuba", "Percussion", "Timpani", "Celesta",
    "Soprano", "Mezzo-Soprano", "Alto", "Tenor",
    "Baritone", "Bass", "Voice", "Choir",
]

# GM MIDI program numbers matching INSTR_NAMES order.
_INSTR_MIDI = [
    0, 0, 6, 19, 21, 25, 33, 40, 41, 42, 43, 46, 73, 68, 71, 70, 65,
    56, 60, 57, 58, 0, 47, 8, 52, 52, 52, 52, 52, 52, 52, 52,
]


def events_to_score(events: list[list[int]]):
    """Compound-event rows -> music21 Score.

    Measure numbers are ``bar_field + 1`` (1-based, like the old decoder), so
    the i-th window bar decodes to measure i+1 regardless of what absolute
    measure it came from — callers that merge fills map measures positionally.
    """
    # lane_events[lane][m_num] = [(offset_ql, element), ...]
    lane_events: dict = defaultdict(lambda: defaultdict(list))

    part_instrs: dict[int, tuple[int, int]] = {}    # part_idx -> (instr, clef)
    measure_key_sigs: dict[int, int] = {}           # m_num -> sharps
    measure_time_sigs: dict[int, int] = {}          # m_num -> TIME_SIGS idx
    measure_tempos: dict[int, float] = {}           # m_num -> BPM
    measure_clefs: dict[tuple[int, int], int] = {}  # (part_idx, m_num) -> clef idx
    measure_chord_syms: dict = defaultdict(list)    # m_num -> [(offset, ChordSymbol)]
    bar_measures: set[int] = set()                  # measures opened by a BAR row

    # First pass: group NOTE/GRACE rows into chords by (lane, m_num, pos, dur, type).
    note_groups: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    group_order: list[tuple] = []

    for row in events:
        t = row[F_TYPE]
        m_num = row[F_BAR] + 1

        if t in (PAD_T, SOS_T, EOS_T, FILL_START_T, FILL_END_T, FILL_GEN_T):
            continue
        if t == INSTR_T:
            part_instrs[row[F_TRACK] - 1] = (row[F_VALUE] - 1, row[F_PITCH] - 1)
            continue
        if t == BAR_T:
            bar_measures.add(m_num)
            continue
        if t == KEY_T:
            measure_key_sigs[m_num] = (row[F_VALUE] - 1) - 6
            continue
        if t == TIME_T:
            idx = row[F_VALUE] - 1
            if 0 <= idx < len(TIME_SIGS):
                measure_time_sigs[m_num] = idx
            continue
        if t == TEMPO_T:
            bucket = row[F_VALUE] - 1
            if 0 <= bucket < N_TEMPO:
                lo = TEMPO_BUCKETS[bucket]
                hi = TEMPO_BUCKETS[bucket + 1] if bucket + 1 < N_TEMPO else lo + 40
                measure_tempos[m_num] = (lo + hi) / 2.0
            continue
        if t == CLEF_T:
            clef_idx = row[F_VALUE] - 1
            if 0 <= clef_idx < N_CLEFS:
                measure_clefs[(row[F_TRACK] - 1, m_num)] = clef_idx
            continue
        if t == CHORD_T:
            bass = row[F_VALUE] - 1 if row[F_VALUE] > 0 else None
            cs = make_chordsym(row[F_PITCH] - 1, row[F_DUR] - 1, bass)
            if cs is not None:
                measure_chord_syms[m_num].append((row[F_POS] * STEP_SIZE, cs))
            continue
        if t == REST_T:
            lane = row[F_TRACK] - 1
            dur = DURATIONS[min(max(row[F_DUR] - 1, 0), N_DURATIONS - 1)]
            lane_events[lane][m_num].append(
                (row[F_POS] * STEP_SIZE, m21.note.Rest(quarterLength=dur)))
            continue
        if t in (NOTE_T, GRACE_T):
            lane = row[F_TRACK] - 1
            key = (lane, m_num, row[F_POS], row[F_DUR], t)
            if key not in note_groups:
                group_order.append(key)
            note_groups[key].append((row[F_PITCH] - 1, row[F_VALUE]))
            continue

    # Materialise grouped notes/chords.
    for key in group_order:
        lane, m_num, pos, dur_field, t = key
        pitches = note_groups[key]
        dur = DURATIONS[min(max(dur_field - 1, 0), N_DURATIONS - 1)]
        midi_list = [max(0, min(127, p)) for p, _ in pitches]
        if len(midi_list) == 1:
            el = m21.note.Note(midi=midi_list[0], quarterLength=dur)
        else:
            el = m21.chord.Chord(
                [m21.pitch.Pitch(midi=p) for p in midi_list], quarterLength=dur)
        if t == GRACE_T:
            el = el.getGrace()
        elif all(tie for _, tie in pitches) and pitches:
            el.tie = m21.tie.Tie('start')
        lane_events[lane][m_num].append((pos * STEP_SIZE, el))

    # Connect tie starts to the next same-pitch note within the same lane.
    for lane, m_dict in lane_events.items():
        flat: list = []
        for m_num in sorted(m_dict.keys()):
            flat.extend(el for _, el in sorted(m_dict[m_num], key=lambda x: x[0]))
        for idx, el in enumerate(flat):
            if getattr(el, 'tie', None) is None or el.tie.type != 'start':
                continue
            el_midi = (sorted(p.midi for p in el.pitches) if el.isChord
                       else [el.pitch.midi] if el.isNote else [])
            for next_el in flat[idx + 1:]:
                if not (next_el.isNote or next_el.isChord):
                    continue
                next_midi = (sorted(p.midi for p in next_el.pitches)
                             if next_el.isChord else [next_el.pitch.midi])
                if next_midi == el_midi:
                    next_el.tie = m21.tie.Tie('stop')
                    break

    # Group lanes by real part index.
    parts_lanes: dict[int, dict[int, dict]] = defaultdict(dict)
    for lane in sorted(lane_events.keys()):
        part_idx  = lane // MAX_VOICES_PER_PART
        voice_idx = lane % MAX_VOICES_PER_PART
        parts_lanes[part_idx][voice_idx] = lane_events[lane]

    score = m21.stream.Score()
    part_order: list[int] = sorted(parts_lanes.keys())
    for part_idx in part_order:
        part = m21.stream.Part()
        voices_data = parts_lanes[part_idx]

        all_measures: set[int] = set(bar_measures)
        for v_data in voices_data.values():
            all_measures.update(v_data.keys())
        if part_idx == part_order[0]:
            all_measures.update(measure_chord_syms.keys())

        for m_num in sorted(all_measures):
            measure = m21.stream.Measure(number=m_num)

            if m_num in measure_key_sigs:
                measure.insert(0, m21.key.KeySignature(measure_key_sigs[m_num]))
            if m_num in measure_tempos and part_idx == part_order[0]:
                measure.insert(0, m21.tempo.MetronomeMark(number=measure_tempos[m_num]))
            if (part_idx, m_num) in measure_clefs:
                clef_idx = measure_clefs[(part_idx, m_num)]
                if 0 <= clef_idx < len(_CLEF_CLASSES):
                    measure.insert(0, _CLEF_CLASSES[clef_idx]())

            if len(voices_data) > 1:
                for voice_idx in sorted(voices_data.keys()):
                    pairs = voices_data[voice_idx].get(m_num, [])
                    if not pairs:
                        continue
                    voice = m21.stream.Voice(id=str(voice_idx + 1))
                    for offset_ql, el in sorted(pairs, key=lambda x: x[0]):
                        voice.insert(offset_ql, el)
                    measure.insert(0, voice)
            elif voices_data:
                voice_idx = next(iter(voices_data))
                pairs = voices_data[voice_idx].get(m_num, [])
                for offset_ql, el in sorted(pairs, key=lambda x: x[0]):
                    measure.insert(offset_ql, el)

            # Chord symbols belong to the top staff.
            if part_idx == part_order[0]:
                for offset_ql, cs in measure_chord_syms.get(m_num, []):
                    measure.insert(offset_ql, cs)

            part.append(measure)
        score.append(part)

    if not score.parts:
        part = m21.stream.Part()
        part.append(m21.stream.Measure())
        score.append(part)

    # Reconstruct time signatures (inside the measure, so re-encoding and
    # makeBeams both see them).
    for part in score.parts:
        for measure in part.getElementsByClass(m21.stream.Measure):
            m_num = measure.number if measure.number is not None else 0
            if m_num in measure_time_sigs:
                measure.insert(0, m21.meter.TimeSignature(
                    TIME_SIGS[measure_time_sigs[m_num]]))

    # Reconstruct instruments and opening clefs from the header INSTR events.
    if part_instrs:
        for pos_idx, part in enumerate(score.parts):
            part_idx = part_order[pos_idx] if pos_idx < len(part_order) else pos_idx
            if part_idx not in part_instrs:
                continue
            measures = list(part.getElementsByClass(m21.stream.Measure))
            if not measures:
                continue
            instrument_idx, clef_idx = part_instrs[part_idx]
            if 0 <= instrument_idx < N_INSTR:
                instrument = m21.instrument.Instrument(
                    instrumentName=_INSTR_DISPLAY[instrument_idx])
                instrument.midiProgram = _INSTR_MIDI[instrument_idx]
                measures[0].insert(0, instrument)
            if 0 <= clef_idx < len(_CLEF_CLASSES):
                measures[0].insert(0, _CLEF_CLASSES[clef_idx]())

    try:
        for part in score.parts:
            for measure in part.getElementsByClass(m21.stream.Measure):
                measure.makeBeams(inPlace=True)
            part.makeAccidentals(inPlace=True, overrideStatus=True)
    except Exception as e:
        print(f"makeBeams/makeAccidentals error: {e}")

    return score


# ---------------------------------------------------------------------------
# Gap detection (unchanged from the flat-token tokenizer)
# ---------------------------------------------------------------------------

def identify_empty_measures(score) -> list[tuple[int, int]]:
    """Detect empty measures across all parts.

    A measure is considered empty if ALL parts have zero notes AND zero rests
    in that measure (a single hidden full-measure rest counts as empty).
    Returns (start_measure, end_measure) tuples for contiguous empty regions.
    """
    parts = list(score.parts)[:MAX_REAL_PARTS]
    if not parts:
        return []

    all_measure_nums: set[int] = set()
    for part in parts:
        for measure in part.getElementsByClass(m21.stream.Measure):
            m_num = measure.number if measure.number is not None else 0
            all_measure_nums.add(m_num)

    if not all_measure_nums:
        return []

    empty_measures: list[int] = []
    for m_num in sorted(all_measure_nums):
        is_empty = True
        for part in parts:
            measures = [m for m in part.getElementsByClass(m21.stream.Measure)
                        if (m.number if m.number is not None else 0) == m_num]
            if not measures:
                is_empty = False
                break
            measure = measures[0]
            notes_and_rests = list(measure.flatten().notesAndRests)
            if len(notes_and_rests) == 0:
                continue
            if len(notes_and_rests) != 1:
                is_empty = False
                break
            el = notes_and_rests[0]
            if not el.isRest:
                is_empty = False
                break
            if not (hasattr(el, 'style') and el.style.hideObjectOnPrint):
                is_empty = False
                break
        if is_empty:
            empty_measures.append(m_num)

    if not empty_measures:
        return []

    gaps: list[tuple[int, int]] = []
    start = end = empty_measures[0]
    for m_num in empty_measures[1:]:
        if m_num == end + 1:
            end = m_num
        else:
            gaps.append((start, end))
            start = end = m_num
    gaps.append((start, end))
    return gaps


def count_bars(events: list[list[int]]) -> int:
    """Number of BAR events in a sequence (measures generated)."""
    return sum(1 for r in events if r[F_TYPE] == BAR_T)


def score_to_xml_str(score) -> str:
    from music21.musicxml.m21ToXml import ScoreExporter
    import xml.etree.ElementTree as ET
    return ET.tostring(ScoreExporter(score).parse(), encoding="utf-8").decode("utf-8")
