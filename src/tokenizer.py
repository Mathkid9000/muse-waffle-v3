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
  240-247  PART_0 … PART_7  (staff/part context switch)
  248-295  STEP_0 … STEP_47 (absolute offset within measure, 32nd-note units)

Within each measure events are grouped first by timestep then by part:

  [BAR]
    [STEP_0]
      [PART_0] [VEL] [NOTE] [DUR]        ← voice 1 of staff 0
               [VEL] [NOTE] [DUR]        ← voice 2 of staff 0 (same STEP+PART)
      [PART_1] [VEL] [NOTE] [DUR]        ← staff 1
    [STEP_4]
      [PART_0] [VEL] [NOTE] [DUR]
    ...

Multiple VEL-NOTE-DUR groups under the same STEP+PART represent concurrent
voices; the model learns to produce them at the correct timestep.
"""

from __future__ import annotations

DURATIONS = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0,
             2.25, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 32.0]

TIME_SIGS = ["2/4", "3/4", "4/4", "6/8", "3/8", "12/8"]

MAX_PARTS  = 8
MAX_STEP   = 48   # 48 × 0.125 ql = 6 quarter notes; covers up to 12/8

PAD, BOS, EOS, BAR = 0, 1, 2, 3
NOTE_OFF   = 4
DUR_OFF    = 132
REST_OFF   = 148
KEY_OFF    = 164
TIME_OFF   = 177
TEXT_OFF   = 183
PART_OFF   = 240
STEP_OFF   = 248
VOCAB_SIZE = STEP_OFF + MAX_STEP   # 296


def _closest_dur_idx(qlen: float) -> int:
    return min(range(len(DURATIONS)), key=lambda i: abs(DURATIONS[i] - qlen))



def _offset_to_step(offset: float) -> int:
    """Quarter-length offset → 32nd-note step index, clamped to [0, MAX_STEP)."""
    return min(MAX_STEP - 1, max(0, round(offset / 0.125)))


def score_to_tokens(score) -> list[int]:
    """music21 Score → timestep-structured token list.

    Events are sorted by (measure, offset, part_idx).  A STEP token is emitted
    whenever the offset changes within a measure; a PART token is emitted
    whenever the active part changes within a timestep.  Multiple voice events
    for the same part at the same offset are written consecutively under the
    same STEP+PART header, producing natural voice stacking.
    """
    import music21 as m21

    parts = list(score.parts)[:MAX_PARTS]
    if not parts:
        return [BOS, EOS]

    # (measure_number, offset_in_measure, part_idx, token_list)
    events: list[tuple[int, float, int, list[int]]] = []

    for part_idx, part in enumerate(parts):
        for measure in part.getElementsByClass(m21.stream.Measure):
            m_num = measure.number if measure.number is not None else 0

            # Yield each voice as a separate stream; fall back to the measure
            # itself when there are no explicit voice containers.
            streams = (
                [v.notesAndRests for v in measure.voices]
                if measure.voices
                else [measure.notesAndRests]
            )

            for note_stream in streams:
                for el in note_stream:
                    offset   = float(el.offset)
                    dur_idx  = _closest_dur_idx(el.duration.quarterLength)

                    if el.isRest:
                        toks = [REST_OFF + dur_idx]
                    elif el.isNote or el.isChord:
                        note_list = [el] if el.isNote else list(el.notes)
                        toks = [NOTE_OFF + n.pitch.midi for n in note_list]
                        toks.append(DUR_OFF + dur_idx)
                    else:
                        continue

                    events.append((m_num, offset, part_idx, toks))

    # Primary sort: measure → offset → part index
    events.sort(key=lambda e: (e[0], e[1], e[2]))

    tokens: list[int] = [BOS]
    cur_measure = -1
    cur_offset  = -1.0
    cur_part    = -1

    for m_num, offset, part_idx, toks in events:
        if m_num != cur_measure:
            tokens.append(BAR)
            cur_measure = m_num
            cur_offset  = -1.0
            cur_part    = -1

        if offset != cur_offset:
            tokens.append(STEP_OFF + _offset_to_step(offset))
            cur_offset = offset
            cur_part   = -1     # re-emit PART after every new STEP

        if part_idx != cur_part:
            tokens.append(PART_OFF + part_idx)
            cur_part = part_idx

        tokens.extend(toks)

    tokens.append(EOS)
    return tokens


def tokens_to_score(tokens: list[int]):
    """Timestep token list → music21 Score.

    Elements that share the same (part, measure, offset) are simultaneous
    voices and are placed in separate Voice containers so notation software
    renders them stacked rather than sequential.
    """
    import music21 as m21
    from collections import defaultdict

    # parts_events[part_idx][m_num][offset_ql] = [el, el, ...]
    parts_events: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    cur_part    = 0
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
            cur_part = t - PART_OFF
            i += 1
            continue

        if REST_OFF <= t < REST_OFF + 16:
            dur = DURATIONS[t - REST_OFF]
            parts_events[cur_part][cur_measure][cur_offset].append(
                m21.note.Rest(quarterLength=dur)
            )
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
            parts_events[cur_part][cur_measure][cur_offset].append(el)
            continue

        i += 1

    # Build score
    score = m21.stream.Score()
    for part_idx in sorted(parts_events.keys()):
        part = m21.stream.Part()
        for m_num in sorted(parts_events[part_idx].keys()):
            measure  = m21.stream.Measure(number=m_num)
            off_map  = parts_events[part_idx][m_num]   # offset → [elements]

            # If any offset carries more than one element, the measure has
            # simultaneous voices and needs Voice containers throughout.
            max_voices = max(len(els) for els in off_map.values())

            if max_voices > 1:
                voices = [m21.stream.Voice(id=str(v + 1))
                          for v in range(max_voices)]
                for offset_ql in sorted(off_map.keys()):
                    for v_idx, el in enumerate(off_map[offset_ql]):
                        voices[v_idx].insert(offset_ql, el)
                for voice in voices:
                    measure.insert(0, voice)
            else:
                for offset_ql in sorted(off_map.keys()):
                    measure.insert(offset_ql, off_map[offset_ql][0])

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
