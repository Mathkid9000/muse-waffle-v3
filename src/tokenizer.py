"""
Converts music21 scores to integer token sequences and back.

Vocabulary layout:
  0        PAD
  1        BOS
  2        EOS
  3        BAR
  4        TIE   (note is tied forward to the next same-pitch note)
  5-132    NOTE  pitch 0-127  (MIDI)
  133-153  DURATION  (21 bins: 32nd … double-whole, plus triplet 16th/8th/quarter/half/whole)
  154-174  REST      (same 21 duration bins)
  175-187  KEY_SIG   (-6 … +6 sharps)
  188-193  TIME_SIG  (common meters)
  194-250  TEXT chars (printable ASCII 32-88, 57 values)
  251-266  PART_0 … PART_15  (combined part+voice: part_idx*2 + voice_idx)
  267-410  STEP_0 … STEP_143 (absolute offset within measure, 1/24 ql units)
  411      GRACE  (next note/chord is a grace note; DUR carries its written duration)

PART tokens encode both staff and voice: PART_{part*2 + voice}.
Each token unambiguously identifies a single voice lane.  Supports up to
8 staves × 2 voices = 16 lanes.

Step resolution is 1/24 ql, which divides both 32nd-notes (3 steps) and
triplet 16th/8th/quarter notes (4/8/16 steps) exactly, preventing rounding
collisions between regular and tuplet notes that would otherwise silently drop
events.

Within each measure events are emitted in (offset, combined_part_voice) order:

  [BAR]
    [STEP_0]
      [PART_0] [NOTE] [DUR]                ← plain note
      [PART_0] [NOTE] [DUR] [TIE]         ← tied note
      [PART_0] [GRACE] [NOTE] [DUR]       ← grace note (written dur in DUR)
      [PART_2] [NOTE] [DUR]                ← part 1, voice 0
    [STEP_8]                                ← offset 8/24 = 0.333 ql (triplet 8th)
      [PART_0] [NOTE] [DUR]
"""

from __future__ import annotations

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

MAX_REAL_PARTS      = 8
MAX_VOICES_PER_PART = 2
MAX_PARTS           = MAX_REAL_PARTS * MAX_VOICES_PER_PART  # 16

# Step resolution: 1/24 ql per step.
# 1/24 is the LCM of 1/8 (32nd notes) and 1/3 (triplets), so both quantise
# to integers with no rounding error.  144 steps × (1/24 ql) = 6 ql, matching
# the previous 48-step range while covering 12/8 time signatures fully.
STEP_SIZE = 1 / 24
MAX_STEP  = 144

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
TEXT_OFF = TIME_OFF + 6              # 194  (ASCII 32-88 = 57 chars)
PART_OFF = TEXT_OFF + 57             # 251
STEP_OFF = PART_OFF + MAX_PARTS      # 267
GRACE    = STEP_OFF + MAX_STEP       # 411

# Infill special tokens (for fill-in-the-middle generation)
FILL_START = GRACE + 1               # 412 - marks start of fill region (prefix ends here)
FILL_END   = FILL_START + 1          # 413 - marks end of suffix, start of generation
EMPTY_BAR  = FILL_END + 1            # 414 - placeholder for empty measures in input

VOCAB_SIZE = EMPTY_BAR + 1           # 415


def _closest_dur_idx(qlen: float) -> int:
    return min(range(len(DURATIONS)), key=lambda i: abs(DURATIONS[i] - qlen))


def _offset_to_step(offset: float) -> int:
    """Quarter-length offset → 1/24-ql step index, clamped to [0, MAX_STEP)."""
    return min(MAX_STEP - 1, max(0, round(offset / STEP_SIZE)))


def score_to_tokens(score) -> list[int]:
    """music21 Score → timestep-structured token list.

    Events are sorted by (measure, offset, combined_idx).  A STEP token is
    emitted when the offset changes; a PART token when the lane changes.  A TIE
    token is appended after NOTE+DUR when the note carries a tie start/continue,
    allowing the decoder to reconstruct tie connections.
    """
    import music21 as m21

    parts = list(score.parts)[:MAX_REAL_PARTS]
    if not parts:
        return [BOS, EOS]

    events: list[tuple[int, float, int, list[int]]] = []

    for part_idx, part in enumerate(parts):
        for measure in part.getElementsByClass(m21.stream.Measure):
            m_num = measure.number if measure.number is not None else 0

            if measure.voices:
                voice_streams = [
                    (vi, v.notesAndRests)
                    for vi, v in enumerate(measure.voices)
                    if vi < MAX_VOICES_PER_PART
                ]
            else:
                voice_streams = [(0, measure.notesAndRests)]

            for voice_idx, note_stream in voice_streams:
                combined = part_idx * MAX_VOICES_PER_PART + voice_idx
                for el in note_stream:
                    offset    = float(el.offset)
                    is_grace  = el.duration.quarterLength == 0

                    if is_grace:
                        # Use written (non-zero) duration for the DUR token so the
                        # decoder can reconstruct the grace note's notated type.
                        dtype = el.duration.type
                        written_ql = (m21.duration.Duration(dtype).quarterLength
                                      if dtype and dtype != 'zero' else 0.25)
                        dur_idx = _closest_dur_idx(written_ql if written_ql > 0 else 0.25)
                    else:
                        dur_idx = _closest_dur_idx(el.duration.quarterLength)

                    if el.isRest:
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

                    events.append((m_num, offset, combined, toks))

    events.sort(key=lambda e: (e[0], e[1], e[2]))

    tokens: list[int] = [BOS]
    cur_measure = -1
    cur_offset  = -1.0
    cur_lane    = -1

    for m_num, offset, combined, toks in events:
        if m_num != cur_measure:
            tokens.append(BAR)
            cur_measure = m_num
            cur_offset  = -1.0
            cur_lane    = -1

        if offset != cur_offset:
            tokens.append(STEP_OFF + _offset_to_step(offset))
            cur_offset = offset
            cur_lane   = -1

        if combined != cur_lane:
            tokens.append(PART_OFF + combined)
            cur_lane = combined

        tokens.extend(toks)

    tokens.append(EOS)
    return tokens


def tokens_to_score(tokens: list[int]):
    """Timestep token list → music21 Score.

    lane_events[combined][m_num] stores a list of (offset_ql, element) pairs.
    Using a list (rather than a dict keyed by offset) prevents silent overwrites
    when two events in the same lane round to the same quantised step — e.g. a
    triplet 8th at 0.333 ql and a dotted 8th at 0.375 ql both hitting step 8
    before the step-resolution upgrade.
    """
    import music21 as m21
    from collections import defaultdict

    # lane_events[combined][m_num] = [(offset_ql, element), ...]
    lane_events: dict = defaultdict(lambda: defaultdict(list))

    cur_lane      = 0
    cur_measure   = 0
    cur_offset    = 0.0
    next_is_grace = False

    i = 0
    while i < len(tokens):
        t = tokens[i]

        if t in (BOS, EOS, PAD):
            i += 1
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

    return score


def text_to_tokens(text: str) -> list[int]:
    """ASCII text → token ids (printable chars in range 32-88 → TEXT_OFF+0..56)."""
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
    import music21 as m21

    parts = list(score.parts)[:MAX_REAL_PARTS]
    print(parts)
    if not parts:
        return []

    # Collect all measure numbers across all parts
    all_measure_nums: set[int] = set()
    for part in parts:
        for measure in part.getElementsByClass(m21.stream.Measure):
            m_num = measure.number if measure.number is not None else 0
            all_measure_nums.add(m_num)
            
    print(all_measure_nums)

    if not all_measure_nums:
        return []

    # Check each measure: empty if ALL parts have no notes/rests
    empty_measures: list[int] = []
    for m_num in sorted(all_measure_nums):
        is_empty = True
        for part in parts:
            measures = [m for m in part.getElementsByClass(m21.stream.Measure)
                       if (m.number if m.number is not None else 0) == m_num]
            # if measures:
            #     measure = measures[0]
            #     # Check for any notes or rests
            #     notes_and_rests = list(measure.flatten().notesAndRests)
            #     if not (len(notes_and_rests) == 1 and notes_and_rests[0].isRest and notes_and_rests[0].quarterLength == 4):
            #         is_empty = False
            #         break
            if not measures:
                is_empty = False
                break
            measure = measures[0]
            notes_and_rests = list(measure.flatten().notesAndRests)
            if len(notes_and_rests) != 1:
                is_empty = False
                break
            el = notes_and_rests[0]
            print(f"{el}, {el.isRest}, {el.style.hideObjectOnPrint}")
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


def score_to_tokens_by_measure(score) -> dict[int, list[int]]:
    """music21 Score → dict mapping measure number to tokens for that measure.

    Returns tokens for each measure separately (without BOS/EOS).
    Each measure's tokens start with BAR followed by that measure's content.
    """
    import music21 as m21

    parts = list(score.parts)[:MAX_REAL_PARTS]
    if not parts:
        return {}

    # Collect events by measure
    events_by_measure: dict[int, list[tuple[float, int, list[int]]]] = {}

    for part_idx, part in enumerate(parts):
        for measure in part.getElementsByClass(m21.stream.Measure):
            m_num = measure.number if measure.number is not None else 0

            if m_num not in events_by_measure:
                events_by_measure[m_num] = []

            if measure.voices:
                voice_streams = [
                    (vi, v.notesAndRests)
                    for vi, v in enumerate(measure.voices)
                    if vi < MAX_VOICES_PER_PART
                ]
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

                    if el.isRest:
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

    # Convert events to tokens per measure
    measure_tokens: dict[int, list[int]] = {}

    for m_num in sorted(events_by_measure.keys()):
        events = sorted(events_by_measure[m_num], key=lambda e: (e[0], e[1]))
        tokens: list[int] = [BAR]
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
    score, gap_start: int, gap_end: int
) -> tuple[list[int], list[int]]:
    """Extract prefix and suffix tokens around a gap region.

    Args:
        score: music21 Score object
        gap_start: First measure number of the gap (inclusive)
        gap_end: Last measure number of the gap (inclusive)

    Returns:
        (prefix_tokens, suffix_tokens)
        - prefix_tokens: Tokens for all measures before the gap (without BOS/EOS)
        - suffix_tokens: Tokens for all measures after the gap (without BOS/EOS)
    """
    measure_tokens = score_to_tokens_by_measure(score)

    prefix: list[int] = []
    suffix: list[int] = []

    for m_num in sorted(measure_tokens.keys()):
        if m_num < gap_start:
            prefix.extend(measure_tokens[m_num])
        elif m_num > gap_end:
            suffix.extend(measure_tokens[m_num])

    return prefix, suffix


def get_gap_measure_count(tokens: list[int]) -> int:
    """Count the number of BAR tokens in a sequence (measures generated)."""
    return tokens.count(BAR)
