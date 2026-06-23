"""
Converts music21 scores to integer token sequences and back.

Vocabulary layout: (471 tokens)
  0        PAD
  1        BOS
  2        EOS
  3        BAR
  4        TIE   (note is tied forward to the next same-pitch note)
  5-132    NOTE  pitch 0-127  (MIDI)
  133-153  DURATION  (21 bins: 32nd to double-whole, plus triplet 16th/8th/quarter/half/whole)
  154-174  REST      (same 21 duration bins)
  175-187  KEY_SIG   (-6 to +6 sharps)
  188-193  TIME_SIG  (common meters)
  194-200  CLEF
  201-257  TEXT chars (printable ASCII 32-88, 57 values)
  258-273  PART_0 to PART_15  (combined part+voice: part_idx*2 + voice_idx)
  274-417  STEP_0 to STEP_143 (absolute offset within measure, 1/24 ql units)
  418      GRACE  (next note/chord is a grace note; DUR carries its written duration)
  419      FILL_START  (infill: end of prefix)
  420      FILL_END    (infill: end of suffix; start of kept-staff gap context)
  421      EMPTY_BAR   (infill: placeholder for empty measures)
  422-437  TEMPO  (16 BPM buckets)
  438-469  INSTR  (32 instrument categories; see INSTR_NAMES)
  470      FILL_GEN    (infill: end of kept-staff gap context; start of generation)

Infill (fill-in-the-middle) sequence format:
  [BOS] header prefix [FILL_START] suffix [FILL_END] kept_gap [FILL_GEN] target_gap [EOS]
  where kept_gap holds the gap measures restricted to the staves being KEPT (context),
  and target_gap holds the gap measures restricted to the staves being regenerated
  (the loss/generation region). For full-staff infill kept_gap is empty.

PART tokens encode both staff and voice: PART_{part*2 + voice}.
Each token unambiguously identifies a single voice lane.  Supports up to
8 staves x 2 voices = 16 lanes.

Step resolution is 1/24 ql per step.
1/24 is the LCM of 1/8 (32nd notes) and 1/3 (triplets), so both quantise
to integers with no rounding error.  144 steps x (1/24 ql) = 6 ql, matching
the previous 48-step range while covering 12/8 time signatures fully.

Each measure opens with BAR followed by optional global symbols, then note events:

  [BAR] [KEY_SIG?] [TIME_SIG?] [TEMPO?]   <- only emitted when present in that measure
    [STEP_0]
      [PART_0] [NOTE] [DUR]
      [PART_0] [NOTE] [DUR] [TIE]
      [PART_0] [GRACE] [NOTE] [DUR]
      [PART_2] [NOTE] [DUR]
    [STEP_8]
      [PART_0] [NOTE] [DUR]

Instrument tokens appear once per part right after BOS (before the first BAR):
  [BOS] [INSTR_0 INSTR_1 ...] [BAR] ...
"""

from __future__ import annotations
import music21 as m21
from collections import defaultdict

# 21 duration bins (quarter-lengths), sorted ascending.
# Includes common triplet values (1/6, 1/3, 2/3, 4/3, 8/3) so tuplet notes
# are quantized exactly instead of being rounded to the nearest regular value.
# 1/6 = triplet 16th, 1/3 = triplet 8th, 2/3 = triplet quarter, etc.
DURATIONS = [
    0.125,  1/6,    0.25,   1/3,
    0.375,  0.5,    2/3,    0.75,
    1.0,    4/3,    1.5,    2.0,
    2.25,   8/3,    3.0,    4.0,
    6.0,    8.0,   12.0,   16.0,
    32.0,
]

TIME_SIGS = ["2/4", "3/4", "4/4", "6/8", "3/8", "12/8"]

CLEFS = [ "treble", "bass", "alto", "tenor", "treble8vb", "treble8va", "bass8vb" ]

MAX_REAL_PARTS      = 8
MAX_VOICES_PER_PART = 2
MAX_PARTS           = MAX_REAL_PARTS * MAX_VOICES_PER_PART  # 16

# Step resolution: 1/24 ql per step.
STEP_SIZE = 1 / 24
MAX_STEP  = 144

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

PAD      = 0
BOS      = 1
EOS      = 2
BAR      = 3
TIE      = 4
NOTE_OFF = 5
DUR_OFF  = NOTE_OFF + 128            # 133
REST_OFF = DUR_OFF  + len(DURATIONS) # 154
KEY_OFF  = REST_OFF + len(DURATIONS) # 175
TIME_OFF = KEY_OFF  + 13             # 188
CLEF_OFF = TIME_OFF + 6              # 194
TEXT_OFF = CLEF_OFF + 7              # 198  (ASCII 32-88 = 57 chars)
PART_OFF = TEXT_OFF + 57             # 255
STEP_OFF = PART_OFF + MAX_PARTS      # 271
GRACE    = STEP_OFF + MAX_STEP       # 415

# Infill special tokens (for fill-in-the-middle generation)
FILL_START = GRACE + 1               # 416 - marks start of fill region (prefix ends here)
FILL_END   = FILL_START + 1          # 417 - marks end of suffix, start of generation
EMPTY_BAR  = FILL_END + 1            # 418 - placeholder for empty measures in input

TEMPO_OFF = EMPTY_BAR + 1            # 422
INSTR_OFF = TEMPO_OFF + N_TEMPO      # 438

# FILL_GEN is appended at the END of the vocab so adding it does not shift any
# existing token id (only one new embedding/head row).  It separates the kept-staff
# gap context from the generation region in the FIM format.
FILL_GEN  = INSTR_OFF + N_INSTR      # 470 - end of kept-staff gap context, start of generation
VOCAB_SIZE = FILL_GEN + 1            # 471


def _closest_dur_idx(qlen: float) -> int:
    return min(range(len(DURATIONS)), key=lambda i: abs(DURATIONS[i] - qlen))


def _offset_to_step(offset: float) -> int:
    """Quarter-length offset -> 1/24-ql step index, clamped to [0, MAX_STEP)."""
    return min(MAX_STEP - 1, max(0, round(offset / STEP_SIZE)))


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


def score_to_header_tokens(score) -> list[int]:
    """Returns one INSTR token and one CLEF token per part (in order), to be emitted right after BOS."""
    parts = list(score.parts)[:MAX_REAL_PARTS]
    header_toks = []
    for p in parts:
        header_toks.extend([INSTR_OFF + instrument_to_idx(p), CLEF_OFF + extract_part_clef(p)])

    return header_toks


def _extract_measure_symbols(score):
    """Extract per-measure KEY_SIG, TIME_SIG, CLEF, and TEMPO tokens from the first part.

    Returns three dicts mapping measure_number -> token_id.
    Only measures that actually contain the element are included.
    """

    parts = list(score.parts)
    if not parts:
        return {}, {}, {}, {}

    key_toks: dict[int, int] = {}
    time_toks: dict[int, int] = {}
    clef_toks: dict[(int, int), int] = {}
    tempo_toks: dict[int, int] = {}

    # clefs
    for part_idx, part in enumerate(parts):
        for m_num, measure in enumerate(part.getElementsByClass(m21.stream.Measure)):
            for c in measure.getElementsByClass(m21.clef.Clef):
                if c.name in CLEFS:
                    clef_toks[(part_idx, m_num)] = CLEF_OFF + CLEFS.index(c.name)
                break

    for measure in parts[0].getElementsByClass(m21.stream.Measure):
        m_num = measure.number if measure.number is not None else 0

        for ks in measure.getElementsByClass(m21.key.KeySignature):
            sharps = int(ks.sharps)
            if -6 <= sharps <= 6:
                key_toks[m_num] = KEY_OFF + (sharps + 6)
            break  # one key sig per measure is enough

        for ts in measure.getElementsByClass(m21.meter.TimeSignature):
            ratio = ts.ratioString
            if ratio in TIME_SIGS:
                time_toks[m_num] = TIME_OFF + TIME_SIGS.index(ratio)
            break

        for mm in measure.getElementsByClass(m21.tempo.MetronomeMark):
            if mm.number is not None:
                tempo_toks[m_num] = TEMPO_OFF + bpm_to_tempo_idx(float(mm.number))
            break

    return key_toks, time_toks, clef_toks, tempo_toks


def tokens_to_score(tokens: list[int]):
    """Timestep token list -> music21 Score.

    lane_events[combined][m_num] stores a list of (offset_ql, element) pairs.
    Using a list (rather than a dict keyed by offset) prevents silent overwrites
    when two events in the same lane round to the same quantised step.
    """

    # lane_events[combined][m_num] = [(offset_ql, element), ...]
    lane_events: dict = defaultdict(lambda: defaultdict(list))

    cur_lane      = 0
    cur_measure   = 0
    cur_offset    = 0.0
    next_is_grace = False
    part_instrs: list[(int, int)] = []   # INSTR index per part, in order

    # Accumulate global symbols per measure for score reconstruction
    measure_key_sigs: dict[int, int] = {}    # measure -> sharps value
    measure_time_sigs: dict[int, int] = {}   # measure -> TIME_SIGS index
    measure_clefs: dict[(int, int), int] = {}   # instrument idx + measure -> CLEFS index
    measure_tempos: dict[int, float] = {}    # measure -> BPM

    i = 0
    while i < len(tokens):
        t = tokens[i]

        if t in (BOS, EOS, PAD):
            i += 1
            continue

        # Instrument header tokens
        if INSTR_OFF <= t < INSTR_OFF + N_INSTR:
            part_instrs.append((t - INSTR_OFF, tokens[i + 1] - CLEF_OFF))
            i += 2
            continue

        if t == BAR:
            cur_measure += 1
            cur_offset = 0.0
            i += 1
            continue

        if t == TIE:
            i += 1
            continue

        if t == GRACE:
            next_is_grace = True
            i += 1
            continue

        # Global measure symbols
        if KEY_OFF <= t < KEY_OFF + 13:
            measure_key_sigs[cur_measure] = (t - KEY_OFF) - 6
            i += 1
            continue

        if TIME_OFF <= t < TIME_OFF + len(TIME_SIGS):
            measure_time_sigs[cur_measure] = t - TIME_OFF
            i += 1
            continue

        if TEMPO_OFF <= t < TEMPO_OFF + N_TEMPO:
            bucket = t - TEMPO_OFF
            lo = TEMPO_BUCKETS[bucket]
            hi = TEMPO_BUCKETS[bucket + 1] if bucket + 1 < N_TEMPO else lo + 40
            measure_tempos[cur_measure] = (lo + hi) / 2.0
            i += 1
            continue

        if STEP_OFF <= t < STEP_OFF + MAX_STEP:
            cur_offset = (t - STEP_OFF) * STEP_SIZE
            i += 1
            continue

        if PART_OFF <= t < PART_OFF + MAX_PARTS:
            cur_lane = t - PART_OFF
            i += 1
            continue

        if REST_OFF <= t < REST_OFF + len(DURATIONS):
            dur = DURATIONS[t - REST_OFF]
            lane_events[cur_lane][cur_measure].append(
                (cur_offset, m21.note.Rest(quarterLength=dur))
            )
            i += 1
            continue

        if NOTE_OFF <= t < NOTE_OFF + 128:
            pitches = []
            while i < len(tokens) and NOTE_OFF <= tokens[i] < NOTE_OFF + 128:
                pitches.append(tokens[i] - NOTE_OFF)
                i += 1
            dur_idx = 7  # default: quarter note (DURATIONS[7] = 1.0)
            if i < len(tokens) and DUR_OFF <= tokens[i] < DUR_OFF + len(DURATIONS):
                dur_idx = tokens[i] - DUR_OFF
                i += 1
            dur = DURATIONS[dur_idx]
            if len(pitches) == 1:
                el = m21.note.Note(midi=pitches[0], quarterLength=dur)
            else:
                el = m21.chord.Chord(
                    [m21.pitch.Pitch(midi=p) for p in pitches], quarterLength=dur
                )
            if next_is_grace:
                el = el.getGrace()
                next_is_grace = False
            elif i < len(tokens) and tokens[i] == TIE:
                el.tie = m21.tie.Tie('start')
                i += 1
            lane_events[cur_lane][cur_measure].append((cur_offset, el))
            continue

        i += 1

    # Connect tie starts to the next same-pitch note within the same lane.
    for combined, m_dict in lane_events.items():
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
    for combined in sorted(lane_events.keys()):
        part_idx  = combined // MAX_VOICES_PER_PART
        voice_idx = combined % MAX_VOICES_PER_PART
        parts_lanes[part_idx][voice_idx] = lane_events[combined]

    score = m21.stream.Score()
    for part_idx in sorted(parts_lanes.keys()):
        part = m21.stream.Part()
        voices_data = parts_lanes[part_idx]

        all_measures: set[int] = set()
        for v_data in voices_data.values():
            all_measures.update(v_data.keys())

        for m_num in sorted(all_measures):
            measure = m21.stream.Measure(number=m_num)

            # Reconstruct global measure symbols
            if m_num in measure_key_sigs:
                measure.insert(0, m21.key.KeySignature(measure_key_sigs[m_num]))
            if m_num in measure_tempos and part_idx == 0:
                measure.insert(0, m21.tempo.MetronomeMark(number=measure_tempos[m_num]))
            if (part_idx, m_num) in measure_clefs:
                clef_idx = measure_clefs[(part_idx, m_num)]
                clef_map = [
                    m21.clef.TrebleClef, m21.clef.BassClef, m21.clef.AltoClef,
                    m21.clef.TenorClef, m21.clef.Treble8vbClef, m21.clef.Treble8vaClef,
                    m21.clef.Bass8vbClef,
                ]
                if 0 <= clef_idx < len(clef_map):
                    measure.insert(0, clef_map[clef_idx]())

            if len(voices_data) > 1:
                for voice_idx in sorted(voices_data.keys()):
                    pairs = voices_data[voice_idx].get(m_num, [])
                    if not pairs:
                        continue
                    voice = m21.stream.Voice(id=str(voice_idx + 1))
                    for offset_ql, el in sorted(pairs, key=lambda x: x[0]):
                        voice.insert(offset_ql, el)
                    measure.insert(0, voice)
            else:
                voice_idx = next(iter(voices_data))
                pairs = voices_data[voice_idx].get(m_num, [])
                for offset_ql, el in sorted(pairs, key=lambda x: x[0]):
                    measure.insert(offset_ql, el)

            part.append(measure)
        score.append(part)

    if not score.parts:
        part = m21.stream.Part()
        part.append(m21.stream.Measure())
        score.append(part)

    # Reconstruct time sigs
    for part in score.parts:
        measures = list(part.getElementsByClass(m21.stream.Measure))
        for measure in measures:
            if measure.measureNumber in measure_time_sigs:
                part.insert(measure.offset, m21.meter.TimeSignature(TIME_SIGS[measure_time_sigs[measure.measureNumber]])) # janky hack

    # Reconstruct clefs and instrument names
    _INSTR_DISPLAY = [
        "Part", "Piano", "Harpsichord", "Organ", "Accordion",
        "Guitar", "Bass Guitar", "Violin", "Viola", "Cello",
        "Double Bass", "Harp", "Flute", "Oboe", "Clarinet",
        "Bassoon", "Saxophone", "Trumpet", "Horn", "Trombone",
        "Tuba", "Percussion", "Timpani", "Celesta",
        "Soprano", "Mezzo-Soprano", "Alto", "Tenor",
        "Baritone", "Bass", "Voice", "Choir",
    ]
    # GM MIDI program numbers corresponding to each instrument index (matches INSTR_NAMES order)
    _INSTR_MIDI = [
        0,   # 0  other        → acoustic grand piano
        0,   # 1  piano        → acoustic grand piano
        6,   # 2  harpsichord  → harpsichord
        19,  # 3  organ        → church organ
        21,  # 4  accordion    → accordion
        25,  # 5  guitar       → acoustic guitar (steel)
        33,  # 6  bass_guitar  → electric bass (finger)
        40,  # 7  violin       → violin
        41,  # 8  viola        → viola
        42,  # 9  cello        → cello
        43,  # 10 double_bass  → contrabass
        46,  # 11 harp         → orchestral harp
        73,  # 12 flute        → flute
        68,  # 13 oboe         → oboe
        71,  # 14 clarinet     → clarinet
        70,  # 15 bassoon      → bassoon
        65,  # 16 saxophone    → alto sax
        56,  # 17 trumpet      → trumpet
        60,  # 18 horn         → french horn
        57,  # 19 trombone     → trombone
        58,  # 20 tuba         → tuba
        0,   # 21 percussion   → (uses MIDI channel 9; program ignored)
        47,  # 22 timpani      → timpani
        8,   # 23 celesta      → celesta
        52,  # 24 voice_soprano → choir aahs
        52,  # 25 voice_mezzo  → choir aahs
        52,  # 26 voice_alto   → choir aahs
        52,  # 27 voice_tenor  → choir aahs
        52,  # 28 voice_baritone → choir aahs
        52,  # 29 voice_bass   → choir aahs
        52,  # 30 voice        → choir aahs
        52,  # 31 choir        → choir aahs
    ]

    if len(part_instrs) > 0:
        for part_idx, part in enumerate(score.parts):
            if part_idx >= len(part_instrs): continue
            measure = list(part.getElementsByClass(m21.stream.Measure))[0]
            instrument_idx, clef_idx = part_instrs[part_idx]
            instrument = m21.instrument.Instrument(instrumentName=_INSTR_DISPLAY[instrument_idx])
            instrument.midiProgram = _INSTR_MIDI[instrument_idx]
            measure.insert(0, instrument)

            if clef_idx == 0:
                clef = m21.clef.TrebleClef()
            elif clef_idx == 1:
                clef = m21.clef.BassClef()
            elif clef_idx == 2:
                clef = m21.clef.AltoClef()
            elif clef_idx == 3:
                clef = m21.clef.TenorClef()
            elif clef_idx == 4:
                clef = m21.clef.Treble8vbClef()
            elif clef_idx == 5:
                clef = m21.clef.Treble8vaClef()
            elif clef_idx == 6:
                clef = m21.clef.Bass8vbClef()
            measure.insert(0, clef)
        
    try:
        for part in score.parts:
            for measure in part.getElementsByClass(m21.stream.Measure):
                measure.makeBeams(inPlace=True)
            part.makeAccidentals(inPlace=True, overrideStatus=True)
    except Exception as e:
        print(f"makeBeams/makeAccidentals error: {e}")

    return score


def text_to_tokens(text: str) -> list[int]:
    """ASCII text -> token ids (printable chars in range 32-88 -> TEXT_OFF+0..56)."""
    out = []
    for ch in text[:56]:
        code = ord(ch)
        if 32 <= code <= 88:
            out.append(TEXT_OFF + code - 32)
    return out


def identify_empty_measures(score) -> list[tuple[int, int]]:
    """Detect empty measures across all parts.

    A measure is considered empty if ALL parts have zero notes AND zero rests
    in that measure. Returns a list of (start_measure, end_measure) tuples
    for contiguous empty regions (inclusive).
    """

    parts = list(score.parts)[:MAX_REAL_PARTS]
    if not parts:
        return []

    # Collect all measure numbers across all parts
    all_measure_nums: set[int] = set()
    for part in parts:
        for measure in part.getElementsByClass(m21.stream.Measure):
            m_num = measure.number if measure.number is not None else 0
            all_measure_nums.add(m_num)

    if not all_measure_nums:
        return []

    # Check each measure: empty if ALL parts have no notes/rests
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
                continue  # truly empty measure, still possibly empty across parts
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

    # Group contiguous empty measures into ranges
    if not empty_measures:
        return []

    gaps: list[tuple[int, int]] = []
    start = empty_measures[0]
    end = empty_measures[0]

    for m_num in empty_measures[1:]:
        if m_num == end + 1:
            end = m_num
        else:
            gaps.append((start, end))
            start = m_num
            end = m_num
    gaps.append((start, end))

    return gaps


def _collect_events_by_measure(score):
    """Walk a score once and return raw per-measure events plus global symbols.

    Returns (events_by_measure, key_toks, time_toks, tempo_toks) where
    events_by_measure maps measure_number -> list of (offset, combined_lane, toks)
    and combined_lane = part_idx * MAX_VOICES_PER_PART + voice_idx.

    This is the expensive (music21-walking) half of tokenization.  Splitting it
    out lets callers re-emit tokens for arbitrary part subsets cheaply via
    _emit_measure_tokens, which is what partial-staff FIM relies on.
    """
    parts = list(score.parts)[:MAX_REAL_PARTS]
    if not parts:
        return {}, {}, {}, {}

    events_by_measure: dict[int, list[tuple[float, int, list[int]]]] = {}

    for part_idx, part in enumerate(parts):
        for measure in part.getElementsByClass(m21.stream.Measure):
            m_num = measure.number if measure.number is not None else 0

            if m_num not in events_by_measure:
                events_by_measure[m_num] = []

            if measure.voices:
                voice_streams = []
                found_v = 0
                has_filled_voice = any([ (True if any([ (False if el.isRest else True) for el in v.notesAndRests ]) else False) for v in measure.voices ])
                for v in measure.voices:
                    els = v.notesAndRests
                    if any([ (False if el.isRest else True) for el in els ]) or not has_filled_voice:
                        voice_streams.append((found_v, els))
                        found_v += 1
                        if found_v >= MAX_VOICES_PER_PART: break
            else:
                voice_streams = [(0, measure.notesAndRests)]

            for voice_idx, note_stream in voice_streams:
                combined = part_idx * MAX_VOICES_PER_PART + voice_idx
                for el in note_stream:
                    offset = float(el.offset)
                    is_grace = el.duration.quarterLength == 0

                    if is_grace:
                        dtype = el.duration.type
                        written_ql = (m21.duration.Duration(dtype).quarterLength
                                      if dtype and dtype != 'zero' else 0.25)
                        dur_idx = _closest_dur_idx(written_ql if written_ql > 0 else 0.25)
                    else:
                        dur_idx = _closest_dur_idx(el.duration.quarterLength)

                    if el.isRest and len([ 1 for o,c,t in events_by_measure[m_num] if o == offset and c == combined ]) == 0:
                        toks = [REST_OFF + dur_idx]
                    elif el.isNote or el.isChord:
                        note_list = [el] if el.isNote else list(el.notes)
                        toks = []
                        if is_grace:
                            toks.append(GRACE)
                        toks += [NOTE_OFF + n.pitch.midi for n in note_list]
                        toks.append(DUR_OFF + dur_idx)
                        if not is_grace and el.tie is not None and el.tie.type in ('start', 'continue'):
                            toks.append(TIE)
                    else:
                        continue

                    events_by_measure[m_num].append((offset, combined, toks))

    key_toks, time_toks, _, tempo_toks = _extract_measure_symbols(score)

    return events_by_measure, key_toks, time_toks, tempo_toks


def _emit_measure_tokens(
    events_by_measure, key_toks, time_toks, tempo_toks,
    part_filter: set[int] | None = None,
    measure_nums: list[int] | None = None,
) -> dict[int, list[int]]:
    """Assemble per-measure token lists from collected events (cheap, pure-python).

    Args:
        part_filter: if given, only events whose real part index
            (combined_lane // MAX_VOICES_PER_PART) is in the set are emitted.
            BAR + global symbols are still emitted for every measure, so a measure
            with no events for the filtered staves becomes a bare [BAR] (+ symbols).
        measure_nums: explicit list of measure numbers to emit (in order). Defaults
            to all measures present in events_by_measure (sorted). Numbers absent
            from events_by_measure still produce a bare [BAR] (+ symbols).
    """
    if measure_nums is None:
        measure_nums = sorted(events_by_measure.keys())

    measure_tokens: dict[int, list[int]] = {}

    for m_num in measure_nums:
        events = events_by_measure.get(m_num, [])
        if part_filter is not None:
            events = [e for e in events if (e[1] // MAX_VOICES_PER_PART) in part_filter]
        events = sorted(events, key=lambda e: (e[0], e[1]))

        tokens: list[int] = [BAR]

        # Emit global symbols for this measure
        for sym_dict in (key_toks, time_toks, tempo_toks):
            if m_num in sym_dict:
                tokens.append(sym_dict[m_num])

        cur_offset = -1.0
        cur_lane = -1

        for offset, combined, toks in events:
            if offset != cur_offset:
                tokens.append(STEP_OFF + _offset_to_step(offset))
                cur_offset = offset
                cur_lane = -1

            if combined != cur_lane:
                tokens.append(PART_OFF + combined)
                cur_lane = combined

            tokens.extend(toks)

        measure_tokens[m_num] = tokens

    return measure_tokens


def score_to_tokens_by_measure(
    score, part_filter: set[int] | None = None,
) -> dict[int, list[int]]:
    """music21 Score -> dict mapping measure number to tokens for that measure.

    Returns tokens for each measure separately (without BOS/EOS).
    Each measure's tokens start with BAR followed by optional KEY_SIG, TIME_SIG,
    and TEMPO tokens (only when present), then STEP/PART/note content.

    Args:
        part_filter: optional set of real part indices to keep; events from other
            parts are dropped (BAR + global symbols still emitted).
    """
    events_by_measure, key_toks, time_toks, tempo_toks = _collect_events_by_measure(score)
    if not events_by_measure:
        return {}
    return _emit_measure_tokens(
        events_by_measure, key_toks, time_toks, tempo_toks, part_filter=part_filter,
    )


def gap_measure_tokens(
    score, gap_start: int, gap_end: int, part_filter: set[int] | None = None,
) -> list[int]:
    """Flat token list for gap measures [gap_start, gap_end], restricted to part_filter.

    Always emits a [BAR] (+ global symbols) for every measure in the range, even
    when the filtered staves are empty there, so kept/target gap blocks stay
    measure-aligned and BAR counting at generation time is exact.
    """
    events_by_measure, key_toks, time_toks, tempo_toks = _collect_events_by_measure(score)
    measure_nums = list(range(gap_start, gap_end + 1))
    mt = _emit_measure_tokens(
        events_by_measure, key_toks, time_toks, tempo_toks,
        part_filter=part_filter, measure_nums=measure_nums,
    )
    return [t for m_num in measure_nums for t in mt[m_num]]


def score_to_tokens_with_gaps(score) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenizes a score while identifying empty regions that need filling.

    Returns:
        (tokens, gap_descriptors)
        - tokens: Full token sequence with EMPTY_BAR placeholders for gaps
        - gap_descriptors: List of (start_measure, end_measure) for empty regions
    """
    gaps = identify_empty_measures(score)
    measure_tokens = score_to_tokens_by_measure(score)

    if not measure_tokens:
        return [BOS, EOS], gaps

    # Build token sequence, using EMPTY_BAR for gap measures
    gap_measures: set[int] = set()
    for start, end in gaps:
        for m in range(start, end + 1):
            gap_measures.add(m)

    tokens: list[int] = [BOS]
    for m_num in sorted(measure_tokens.keys()):
        if m_num in gap_measures:
            tokens.append(EMPTY_BAR)
        else:
            tokens.extend(measure_tokens[m_num])
    tokens.append(EOS)

    return tokens, gaps


def extract_prefix_suffix_tokens(
    score, gap_start: int, gap_end: int, max_context_measures: int = 16,
) -> tuple[list[int], list[int]]:
    """Extract prefix and suffix tokens around a gap region.

    Only the nearest `max_context_measures` measures on each side are included,
    matching the window used during FIM training.

    Args:
        score: music21 Score object
        gap_start: First measure number of the gap (inclusive)
        gap_end: Last measure number of the gap (inclusive)
        max_context_measures: Maximum measures to include on each side

    Returns:
        (prefix_tokens, suffix_tokens)
        - prefix_tokens: Tokens for up to max_context_measures before the gap
        - suffix_tokens: Tokens for up to max_context_measures after the gap
    """
    measure_tokens = score_to_tokens_by_measure(score)

    pre_nums  = [m for m in sorted(measure_tokens) if m < gap_start][-max_context_measures:]
    post_nums = [m for m in sorted(measure_tokens) if m > gap_end  ][:max_context_measures]

    prefix = [t for m in pre_nums  for t in measure_tokens[m]]
    suffix = [t for m in post_nums for t in measure_tokens[m]]

    return prefix, suffix


def get_gap_measure_count(tokens: list[int]) -> int:
    """Count the number of BAR tokens in a sequence (measures generated)."""
    return tokens.count(BAR)


def score_to_xml_str(score) -> str:
    from music21.musicxml.m21ToXml import ScoreExporter
    import xml.etree.ElementTree as ET
    return ET.tostring(ScoreExporter(score).parse(), encoding="utf-8").decode("utf-8")
