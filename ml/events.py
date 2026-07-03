"""Compound-event schema for the MMT-BERT music representation.

Every event is a fixed 7-field tuple of small non-negative ints
(following the multi-field token design of the Multitrack Music
Transformer / MMT-BERT, arXiv 2409.00919, extended with the fields the
sheet-music product needs for MusicXML round-trip):

    field 0  type    event type (see *_T constants below)
    field 1  bar     bar index within the sequence window, 0..MAX_BARS-1
    field 2  pos     onset within the bar in 1/24-quarter steps, 0..MAX_POS-1
    field 3  pitch   NOTE/GRACE: midi+1 (1..128); CHORD: root pc+1 (1..12);
                     INSTR: clef idx+1 (1..7); else 0
    field 4  dur     NOTE/GRACE/REST: DURATIONS bin+1 (1..21);
                     CHORD: CHORD_KINDS idx+1 (1..26); else 0
    field 5  track   NOTE/GRACE/REST: combined lane (part*2+voice)+1 (1..16);
                     INSTR/CLEF: part idx+1 (1..8); else 0
    field 6  value   type-dependent extra:
                     NOTE  tie-forward flag (0/1)
                     KEY   sharps+6+1 (1..13, i.e. -6..+6 sharps)
                     TIME  TIME_SIGS idx+1 (1..6)
                     TEMPO TEMPO_BUCKETS idx+1 (1..16)
                     CLEF  CLEFS idx+1 (1..7)
                     INSTR INSTR_NAMES idx+1 (1..32)
                     CHORD bass pc+1 (1..12) for slash chords, 0 = no bass

Position resolution stays at 1/24 quarter-length per step (the LCM of
32nd notes and triplets, so both quantise exactly); the paper's 12 steps
per quarter cannot represent 32nd notes.  MAX_POS = 144 steps = 6 ql,
covering 12/8.

Sequence formats (events, not flat tokens):

  standard:  [SOS] [INSTR x P] [BAR KEY? TIME? TEMPO? CLEF? notes...] ... [EOS]
  FIM:       [SOS] header prefix [FILL_START] suffix [FILL_END]
                   kept_gap [FILL_GEN] target_gap [EOS]

Bar indices are window-relative: a standard example numbers its bars
0..N-1; a FIM example numbers prefix bars 0..P-1, gap bars P..P+G-1
(identical in kept_gap and target_gap, which makes vertical alignment
between kept and regenerated staves explicit), suffix bars P+G.. .
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Field indices
# ---------------------------------------------------------------------------

F_TYPE, F_BAR, F_POS, F_PITCH, F_DUR, F_TRACK, F_VALUE = range(7)
N_FIELDS = 7

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

PAD_T        = 0
SOS_T        = 1
EOS_T        = 2
INSTR_T      = 3   # header: one per part (pitch=clef+1, track=part+1, value=instr+1)
BAR_T        = 4   # opens every measure; a bare BAR is an empty measure
KEY_T        = 5   # value = sharps+6+1
TIME_T       = 6   # value = TIME_SIGS idx+1
TEMPO_T      = 7   # value = TEMPO_BUCKETS idx+1
CLEF_T       = 8   # mid-score clef change (track=part+1, value=clef+1)
NOTE_T       = 9   # value = tie-forward flag
GRACE_T      = 10  # dur = written duration bin
REST_T       = 11
CHORD_T      = 12  # chord symbol: pitch=root pc+1, dur=kind+1, value=bass pc+1|0
FILL_START_T = 13  # FIM: end of prefix
FILL_END_T   = 14  # FIM: end of suffix, start of kept-staff gap context
FILL_GEN_T   = 15  # FIM: end of kept-staff gap context, start of generation

N_TYPES = 16

TYPE_NAMES = [
    "PAD", "SOS", "EOS", "INSTR", "BAR", "KEY", "TIME", "TEMPO", "CLEF",
    "NOTE", "GRACE", "REST", "CHORD", "FILL_START", "FILL_END", "FILL_GEN",
]

# ---------------------------------------------------------------------------
# Musical constants (moved from the old flat-token tokenizer, unchanged)
# ---------------------------------------------------------------------------

# 21 duration bins (quarter-lengths), sorted ascending.  Includes common
# triplet values (1/6, 1/3, 2/3, 4/3, 8/3) so tuplet notes are quantized
# exactly instead of being rounded to the nearest regular value.
DURATIONS = [
    0.125,  1/6,    0.25,   1/3,
    0.375,  0.5,    2/3,    0.75,
    1.0,    4/3,    1.5,    2.0,
    2.25,   8/3,    3.0,    4.0,
    6.0,    8.0,   12.0,   16.0,
    32.0,
]
N_DURATIONS = len(DURATIONS)

# Duration bins with no single notation type (music21 'complex'); a NOTE/REST
# with one of these is exported as tied notes, but a zero-length GRACE cannot
# be split, so grace sampling masks them and grace decoding falls back.
COMPLEX_DUR_BINS = (12,)  # 2.25 ql

TIME_SIGS = ["2/4", "3/4", "4/4", "6/8", "3/8", "12/8"]
# Total quarter-lengths per bar for each TIME_SIGS entry (num * 4/den).
TIME_SIG_QL = [2.0, 3.0, 4.0, 3.0, 1.5, 6.0]

CLEFS = ["treble", "bass", "alto", "tenor", "treble8vb", "treble8va", "bass8vb"]
N_CLEFS = len(CLEFS)

MAX_REAL_PARTS      = 8
MAX_VOICES_PER_PART = 2
MAX_PARTS           = MAX_REAL_PARTS * MAX_VOICES_PER_PART  # 16 lanes

# Position resolution: 1/24 ql per step.
STEP_SIZE = 1 / 24
MAX_POS   = 144

# Bar index range within one sequence window.
MAX_BARS = 256

# Tempo: 16 BPM buckets.  Bucket i covers [TEMPO_BUCKETS[i], TEMPO_BUCKETS[i+1]).
N_TEMPO = 16
TEMPO_BUCKETS = [20, 40, 52, 63, 72, 84, 96, 108, 116, 126, 138, 152, 168, 192, 216, 256]

# Instrument categories (32 entries, index 0 = unknown/other).
N_INSTR = 32
INSTR_NAMES = [
    "other",          # 0
    "piano",          # 1
    "harpsichord",    # 2
    "organ",          # 3
    "accordion",      # 4
    "guitar",         # 5
    "bass_guitar",    # 6
    "violin",         # 7
    "viola",          # 8
    "cello",          # 9
    "double_bass",    # 10
    "harp",           # 11
    "flute",          # 12
    "oboe",           # 13
    "clarinet",       # 14
    "bassoon",        # 15
    "saxophone",      # 16
    "trumpet",        # 17
    "horn",           # 18
    "trombone",       # 19
    "tuba",           # 20
    "percussion",     # 21
    "timpani",        # 22
    "celesta",        # 23
    "voice_soprano",  # 24
    "voice_mezzo",    # 25
    "voice_alto",     # 26
    "voice_tenor",    # 27
    "voice_baritone", # 28
    "voice_bass",     # 29
    "voice",          # 30 (generic)
    "choir",          # 31
]

# Chord qualities (music21/MusicXML chordKind values).
CHORD_KINDS = [
    "major", "minor", "augmented", "diminished",
    "dominant", "major-seventh", "minor-seventh", "diminished-seventh",
    "augmented-seventh", "half-diminished", "major-minor",
    "major-sixth", "minor-sixth",
    "dominant-ninth", "major-ninth", "minor-ninth",
    "dominant-11th", "major-11th", "minor-11th",
    "dominant-13th", "major-13th", "minor-13th",
    "suspended-second", "suspended-fourth",
    "power", "other",
]
N_CHORD_KINDS = len(CHORD_KINDS)

# ---------------------------------------------------------------------------
# Per-field vocabulary sizes (the 7 output-head / embedding sizes)
# ---------------------------------------------------------------------------

FIELD_VOCAB_SIZES = [
    N_TYPES,    # type   16
    MAX_BARS,   # bar    256
    MAX_POS,    # pos    144
    130,        # pitch  0 + midi 1..128 (clef 1..7 for INSTR fits inside)
    27,         # dur    0 + duration bins 1..21 / chord kinds 1..26
    17,         # track  0 + lanes 1..16 (parts 1..8 for INSTR/CLEF fit inside)
    33,         # value  0 + up to 32 (INSTR categories are the widest use)
]

FIELD_NAMES = ["type", "bar", "pos", "pitch", "dur", "track", "value"]

# ---------------------------------------------------------------------------
# Per-type field relevance: FIELD_LOSS_MASK[type][field] == 1 where the field
# carries information for that event type (drives both training-loss masking
# and structural-zero forcing during sampling).
# ---------------------------------------------------------------------------

FIELD_LOSS_MASK = [
    #                 type bar pos pitch dur track value
    [0, 0, 0, 0, 0, 0, 0],  # PAD
    [1, 0, 0, 0, 0, 0, 0],  # SOS
    [1, 0, 0, 0, 0, 0, 0],  # EOS
    [1, 0, 0, 1, 0, 1, 1],  # INSTR (pitch=clef, track=part, value=instr)
    [1, 1, 0, 0, 0, 0, 0],  # BAR
    [1, 1, 0, 0, 0, 0, 1],  # KEY
    [1, 1, 0, 0, 0, 0, 1],  # TIME
    [1, 1, 0, 0, 0, 0, 1],  # TEMPO
    [1, 1, 0, 0, 0, 1, 1],  # CLEF (track=part, value=clef)
    [1, 1, 1, 1, 1, 1, 1],  # NOTE (value=tie)
    [1, 1, 1, 1, 1, 1, 0],  # GRACE
    [1, 1, 1, 0, 1, 1, 0],  # REST
    [1, 1, 1, 1, 1, 0, 1],  # CHORD (pitch=root, dur=kind, value=bass|0)
    [1, 0, 0, 0, 0, 0, 0],  # FILL_START
    [1, 0, 0, 0, 0, 0, 0],  # FILL_END
    [1, 0, 0, 0, 0, 0, 0],  # FILL_GEN
]

# Static per-type valid value ranges for the non-temporal fields, used to mask
# the field heads during sampling.  RANGES[type][field] is an inclusive
# (lo, hi) pair; (0, 0) means the field is structurally forced to 0.  The bar
# and pos fields are handled dynamically by the generation loop (bar is forced
# from BAR-event counting; pos is masked to be monotonic within the bar).
_Z = (0, 0)
FIELD_VALID_RANGES = {
    INSTR_T: {F_PITCH: (1, N_CLEFS), F_DUR: _Z, F_TRACK: (1, MAX_REAL_PARTS), F_VALUE: (1, N_INSTR)},
    BAR_T:   {F_PITCH: _Z, F_DUR: _Z, F_TRACK: _Z, F_VALUE: _Z},
    KEY_T:   {F_PITCH: _Z, F_DUR: _Z, F_TRACK: _Z, F_VALUE: (1, 13)},
    TIME_T:  {F_PITCH: _Z, F_DUR: _Z, F_TRACK: _Z, F_VALUE: (1, len(TIME_SIGS))},
    TEMPO_T: {F_PITCH: _Z, F_DUR: _Z, F_TRACK: _Z, F_VALUE: (1, N_TEMPO)},
    CLEF_T:  {F_PITCH: _Z, F_DUR: _Z, F_TRACK: (1, MAX_REAL_PARTS), F_VALUE: (1, N_CLEFS)},
    NOTE_T:  {F_PITCH: (1, 128), F_DUR: (1, N_DURATIONS), F_TRACK: (1, MAX_PARTS), F_VALUE: (0, 1)},
    GRACE_T: {F_PITCH: (1, 128), F_DUR: (1, N_DURATIONS), F_TRACK: (1, MAX_PARTS), F_VALUE: _Z},
    REST_T:  {F_PITCH: _Z, F_DUR: (1, N_DURATIONS), F_TRACK: (1, MAX_PARTS), F_VALUE: _Z},
    CHORD_T: {F_PITCH: (1, 12), F_DUR: (1, N_CHORD_KINDS), F_TRACK: _Z, F_VALUE: (0, 12)},
}
for _t in (PAD_T, SOS_T, EOS_T, FILL_START_T, FILL_END_T, FILL_GEN_T):
    FIELD_VALID_RANGES[_t] = {F_PITCH: _Z, F_DUR: _Z, F_TRACK: _Z, F_VALUE: _Z}

# Event types that carry an onset (pos is meaningful and sampled).
CONTENT_TYPES = (NOTE_T, GRACE_T, REST_T, CHORD_T)

# ---------------------------------------------------------------------------
# Event constructors
# ---------------------------------------------------------------------------

def special_event(etype: int) -> list[int]:
    return [etype, 0, 0, 0, 0, 0, 0]


def sos_event() -> list[int]:        return special_event(SOS_T)
def eos_event() -> list[int]:        return special_event(EOS_T)
def fill_start_event() -> list[int]: return special_event(FILL_START_T)
def fill_end_event() -> list[int]:   return special_event(FILL_END_T)
def fill_gen_event() -> list[int]:   return special_event(FILL_GEN_T)


def bar_event(bar: int) -> list[int]:
    return [BAR_T, bar, 0, 0, 0, 0, 0]


def key_event(bar: int, sharps: int) -> list[int]:
    return [KEY_T, bar, 0, 0, 0, 0, sharps + 6 + 1]


def time_event(bar: int, ts_idx: int) -> list[int]:
    return [TIME_T, bar, 0, 0, 0, 0, ts_idx + 1]


def tempo_event(bar: int, bucket: int) -> list[int]:
    return [TEMPO_T, bar, 0, 0, 0, 0, bucket + 1]


def clef_event(bar: int, part_idx: int, clef_idx: int) -> list[int]:
    return [CLEF_T, bar, 0, 0, 0, part_idx + 1, clef_idx + 1]


def instr_event(part_idx: int, instr_idx: int, clef_idx: int) -> list[int]:
    return [INSTR_T, 0, 0, clef_idx + 1, 0, part_idx + 1, instr_idx + 1]


def note_event(bar: int, pos: int, midi: int, dur_idx: int, lane: int, tie: bool) -> list[int]:
    return [NOTE_T, bar, pos, midi + 1, dur_idx + 1, lane + 1, 1 if tie else 0]


def grace_event(bar: int, pos: int, midi: int, dur_idx: int, lane: int) -> list[int]:
    return [GRACE_T, bar, pos, midi + 1, dur_idx + 1, lane + 1, 0]


def rest_event(bar: int, pos: int, dur_idx: int, lane: int) -> list[int]:
    return [REST_T, bar, pos, 0, dur_idx + 1, lane + 1, 0]


def chord_event(bar: int, pos: int, root_pc: int, kind_idx: int, bass_pc: int | None) -> list[int]:
    return [CHORD_T, bar, pos, root_pc + 1, kind_idx + 1, 0,
            0 if bass_pc is None else bass_pc + 1]


# ---------------------------------------------------------------------------
# Transposition augmentation
# ---------------------------------------------------------------------------

def transpose_events(events: list[list[int]], semitones: int) -> list[list[int]]:
    """Return transposed copies of `events` (input rows are never mutated).

    NOTE/GRACE pitches shift by `semitones` (clamped to midi 0..127); KEY
    signatures shift via the circle of fifths and wrap into -6..+5 sharps;
    CHORD roots and slash basses rotate mod 12.  All other events copy as-is.
    """
    if semitones == 0:
        return [list(e) for e in events]
    out = []
    for e in events:
        e = list(e)
        t = e[F_TYPE]
        if t in (NOTE_T, GRACE_T):
            e[F_PITCH] = max(1, min(128, e[F_PITCH] + semitones))
        elif t == KEY_T:
            old_sharps = (e[F_VALUE] - 1) - 6
            new_sharps = ((old_sharps + semitones * 7) + 6) % 12 - 6
            e[F_VALUE] = new_sharps + 6 + 1
        elif t == CHORD_T:
            e[F_PITCH] = (e[F_PITCH] - 1 + semitones) % 12 + 1
            if e[F_VALUE] > 0:
                e[F_VALUE] = (e[F_VALUE] - 1 + semitones) % 12 + 1
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# Schema fingerprint inputs (hashed by ml.preprocess.vocab_fingerprint)
# ---------------------------------------------------------------------------

SCHEMA_LAYOUT = (
    "mmt-bert-events",
    N_TYPES, N_FIELDS, tuple(FIELD_VOCAB_SIZES),
    MAX_POS, MAX_BARS, N_DURATIONS, MAX_PARTS, MAX_REAL_PARTS,
    len(TIME_SIGS), N_CLEFS, N_TEMPO, N_INSTR, N_CHORD_KINDS,
)
