"""
Converts music21 scores to integer token sequences and back.

Vocabulary layout:
  0        PAD
  1        BOS
  2        EOS
  3        BAR
  4-131    NOTE pitch 0-127  (MIDI)
  132-147  DURATION  (16 bins: 32nd … double-whole)
  148-163  REST      (same 16 duration bins)
  164-176  KEY_SIG   (-6 … +6 sharps)
  177-182  TIME_SIG  (common meters)
  183-239  TEXT chars (printable ASCII 32-88)
  240-255  PART_0 … PART_15  (combined part+voice: part_idx*2 + voice_idx)
  256-303  STEP_0 … STEP_47  (absolute offset within measure, 32nd-note units)

PART tokens encode both staff and voice: PART_{part*2 + voice}.
Each token unambiguously identifies a single voice lane, so the decoder can
reconstruct Voice containers without guessing which lane a lone element belongs
to.  Supports up to 8 staves × 2 voices = 16 lanes.

Within each measure events are emitted in (offset, combined_part_voice) order:

  [BAR]
    [STEP_0]
      [PART_0] [NOTE] [DUR]   ← part 0, voice 0
      [PART_1] [NOTE] [DUR]   ← part 0, voice 1
      [PART_2] [NOTE] [DUR]   ← part 1, voice 0
    [STEP_4]
      [PART_0] [NOTE] [DUR]
"""

from __future__ import annotations

DURATIONS = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0,
             2.25, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 32.0]

TIME_SIGS = ["2/4", "3/4", "4/4", "6/8", "3/8", "12/8"]

MAX_REAL_PARTS      = 8
MAX_VOICES_PER_PART = 2
MAX_PARTS           = MAX_REAL_PARTS * MAX_VOICES_PER_PART  # 16 combined slots
MAX_STEP            = 48   # 48 × 0.125 ql = 6 quarter notes; covers up to 12/8

PAD, BOS, EOS, BAR = 0, 1, 2, 3
NOTE_OFF   = 4
DUR_OFF    = 132
REST_OFF   = 148
KEY_OFF    = 164
TIME_OFF   = 177
TEXT_OFF   = 183
PART_OFF   = 240
STEP_OFF   = 256   # shifted: 240 + 16 PART slots
VOCAB_SIZE = STEP_OFF + MAX_STEP   # 304


def _closest_dur_idx(qlen: float) -> int:
    return min(range(len(DURATIONS)), key=lambda i: abs(DURATIONS[i] - qlen))



def _offset_to_step(offset: float) -> int:
    """Quarter-length offset → 32nd-note step index, clamped to [0, MAX_STEP)."""
    return min(MAX_STEP - 1, max(0, round(offset / 0.125)))


def score_to_tokens(score) -> list[int]:
    """music21 Score → timestep-structured token list.

    Events are sorted by (measure, offset, combined_idx) where combined_idx
    encodes both part and voice: combined = part_idx * MAX_VOICES_PER_PART +
    voice_idx.  A STEP token is emitted when the offset changes; a PART token
    (carrying combined_idx) is emitted when the lane changes.  Because each
    voice gets its own PART slot, the decoder can reconstruct voice containers
    without ambiguity even when only one voice is active at a given offset.
    """
    import music21 as m21

    parts = list(score.parts)[:MAX_REAL_PARTS]
    if not parts:
        return [BOS, EOS]

    # (measure_number, offset_in_measure, combined_idx, token_list)
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
                    offset  = float(el.offset)
                    dur_idx = _closest_dur_idx(el.duration.quarterLength)

                    if el.isRest:
                        toks = [REST_OFF + dur_idx]
                    elif el.isNote or el.isChord:
                        note_list = [el] if el.isNote else list(el.notes)
                        toks = [NOTE_OFF + n.pitch.midi for n in note_list]
                        toks.append(DUR_OFF + dur_idx)
                    else:
                        continue

                    events.append((m_num, offset, combined, toks))

    # Primary sort: measure → offset → combined (part*voices + voice)
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

    PART tokens encode combined (part_idx, voice_idx) as
    combined = part_idx * MAX_VOICES_PER_PART + voice_idx, so each lane maps
    to an unambiguous Voice container — no positional guessing required.

    lane_events[combined][m_num][offset_ql] = element
    """
    import music21 as m21
    from collections import defaultdict

    # lane_events[combined][m_num][offset_ql] = music21 element
    lane_events: dict = defaultdict(lambda: defaultdict(dict))

    cur_lane    = 0
    cur_measure = 0
    cur_offset  = 0.0

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

        if STEP_OFF <= t < STEP_OFF + MAX_STEP:
            cur_offset = (t - STEP_OFF) * 0.125
            i += 1
            continue

        if PART_OFF <= t < PART_OFF + MAX_PARTS:
            cur_lane = t - PART_OFF
            i += 1
            continue

        if REST_OFF <= t < REST_OFF + 16:
            dur = DURATIONS[t - REST_OFF]
            lane_events[cur_lane][cur_measure][cur_offset] = m21.note.Rest(quarterLength=dur)
            i += 1
            continue

        if NOTE_OFF <= t < NOTE_OFF + 128:
            pitches = []
            while i < len(tokens) and NOTE_OFF <= tokens[i] < NOTE_OFF + 128:
                pitches.append(tokens[i] - NOTE_OFF)
                i += 1
            dur_idx = 5
            if i < len(tokens) and DUR_OFF <= tokens[i] < DUR_OFF + 16:
                dur_idx = tokens[i] - DUR_OFF
                i += 1
            dur = DURATIONS[dur_idx]
            if len(pitches) == 1:
                el = m21.note.Note(midi=pitches[0], quarterLength=dur)
            else:
                el = m21.chord.Chord(
                    [m21.pitch.Pitch(midi=p) for p in pitches], quarterLength=dur
                )
            lane_events[cur_lane][cur_measure][cur_offset] = el
            continue

        i += 1

    # Group lanes by real part index
    # combined = part_idx * MAX_VOICES_PER_PART + voice_idx
    parts_lanes: dict[int, dict[int, dict]] = defaultdict(dict)
    for combined in sorted(lane_events.keys()):
        part_idx  = combined // MAX_VOICES_PER_PART
        voice_idx = combined % MAX_VOICES_PER_PART
        parts_lanes[part_idx][voice_idx] = lane_events[combined]

    score = m21.stream.Score()
    for part_idx in sorted(parts_lanes.keys()):
        part = m21.stream.Part()
        voices_data = parts_lanes[part_idx]  # voice_idx → {m_num → {offset → el}}

        all_measures: set[int] = set()
        for v_data in voices_data.values():
            all_measures.update(v_data.keys())

        for m_num in sorted(all_measures):
            measure = m21.stream.Measure(number=m_num)

            if len(voices_data) > 1:
                # Multiple voice lanes → use Voice containers
                for voice_idx in sorted(voices_data.keys()):
                    off_map = voices_data[voice_idx].get(m_num, {})
                    if not off_map:
                        continue
                    voice = m21.stream.Voice(id=str(voice_idx + 1))
                    for offset_ql in sorted(off_map.keys()):
                        voice.insert(offset_ql, off_map[offset_ql])
                    measure.insert(0, voice)
            else:
                voice_idx = next(iter(voices_data))
                off_map = voices_data[voice_idx].get(m_num, {})
                for offset_ql in sorted(off_map.keys()):
                    measure.insert(offset_ql, off_map[offset_ql])

            part.append(measure)
        score.append(part)

    if not score.parts:
        part = m21.stream.Part()
        part.append(m21.stream.Measure())
        score.append(part)

    return score


def text_to_tokens(text: str) -> list[int]:
    """ASCII text → token ids (printable chars in range 32-88 → 199-255)."""
    out = []
    for ch in text[:56]:
        code = ord(ch)
        if 32 <= code <= 88:
            out.append(TEXT_OFF + code - 32)
    return out
