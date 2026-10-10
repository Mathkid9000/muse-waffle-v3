"""Engraving: the rules that turn (bar, pos, dur, pitch, staff) events into notation that follows
standard conventions, shared by every producer of MusicXML in the project.

The model's rows say *what* sounds and when, on a 24-per-quarter grid, and nothing else: no rests,
no voices, no ties, no spelling, no accidentals. Everything a reader expects to be conventional is
decided here, deterministically, so a score renders the same however its rows arrived:

  * grid snapping and bar bounds   (sanitize_measure, clip_to_bar)
  * voice separation, up to MAX_VOICES per staff, stems by voice   (assign_voices)
  * note splitting/tying at the beats a note may not obscure   (split_note)
  * visible rests, grouped by the metric tree   (partition_rest, voice_elements)
  * every bar complete, a blank bar a whole rest   (fill_rests)
  * enharmonic spelling from key, chord and melodic context   (spell_pitch)
  * accidental display, once per bar, in time order across voices   (make_accidentals)
  * a clef, key, meter or tempo written only where it changes   (drop_restated, carry_state)
  * beams, tuplet brackets, and an export that re-derives nothing   (engrave, export_xml)

Positions and durations in this module are integers in grid units unless a name says otherwise
(24 = a quarter). music21 objects only appear once the rhythm is settled.
"""

from __future__ import annotations
import bisect
import copy
import logging
from dataclasses import dataclass
from fractions import Fraction

import music21
from music21 import chord as chord21
from music21 import clef as clef21
from music21 import duration as duration21
from music21 import harmony as harmony21
from music21 import key as key21
from music21 import meter as meter21
from music21 import note as note21
from music21 import pitch as pitch21
from music21 import stream as stream21
from music21 import tempo as tempo21
from music21 import tie as tie21
from music21.stream import makeNotation

log = logging.getLogger(__name__)

SUBDIVISIONS_PER_QUARTER = 24     # must equal tokenizer.SUBDIVISIONS_PER_QUARTER (asserted there)
WHOLE = 4 * SUBDIVISIONS_PER_QUARTER
MAX_VOICES = 4
STEM_DIRECTIONS = ('up', 'down', 'up', 'down')   # voice 1 up, 2 down, 3 up, 4 down
PLAIN_DURATIONS = {3, 6, 12, 24, 48, 96, 192}    # 32nd .. breve
DOTTED_DURATIONS = {9, 18, 36, 72, 144}
TRIPLET_DURATIONS = (16, 8, 4)                   # triplet quarter, eighth, sixteenth
NEW_VOICE_COST = 8       # semitones: a voice's last note further away than this and a free voice opens


# ---------------------------------------------------------------- meter

def bar_length(num: int, den: int) -> int:
    return num * WHOLE // den


def is_compound(num: int, den: int) -> bool:
    return den >= 8 and num % 3 == 0


def beat_length(num: int, den: int) -> int:
    if is_compound(num, den):
        return 3 * WHOLE // den
    return WHOLE // den


def metric_levels(num: int, den: int) -> list[int]:
    """Node lengths of the bar's metric tree, largest first: the bar, its strong halves where the
    meter has them, the beat, then binary subdivisions down to a 32nd. A boundary at grid offset t
    is as strong as the largest level dividing t; a note or rest may not obscure a boundary
    stronger than itself."""
    bar, beat = bar_length(num, den), beat_length(num, den)
    levels = [bar]
    beats = num // 3 if is_compound(num, den) else num
    if beats % 2 == 0 and beats > 2:
        levels.append(beat * beats // 2)
    levels.append(beat)
    sub = beat // 3 if is_compound(num, den) else beat // 2
    while sub >= 3:
        levels.append(sub)
        sub //= 2
    return sorted(set(levels), reverse=True)


def node_size(t: int, levels: list[int]) -> int:
    return max((lvl for lvl in levels if t % lvl == 0), default=0)


# ---------------------------------------------------------------- events

@dataclass
class NoteIn:
    midi: int
    label: object = None
    tie_stop: bool = False   # continues a note cut at the previous barline
    notehead: str | None = None   # eg. 'slash' for a template's still-unpitched note


@dataclass
class Event:
    """One onset in one voice: a note or a chord, `dur` long, at grid offset `pos` in its bar."""
    pos: int
    dur: int
    notes: list[NoteIn]
    tie_start: bool = False   # cut at the barline; a continuation follows in the next bar
    uid: int = 0              # links a cut note to its continuation, so truncating one drops the other

    @property
    def top(self) -> int:
        return max(n.midi for n in self.notes)

    @property
    def mean(self) -> float:
        return sum(n.midi for n in self.notes) / len(self.notes)

    @property
    def end(self) -> int:
        return self.pos + self.dur


def _snap3(v: int) -> int:
    return max(3, round(v / 3) * 3)


def _snap4(v: int) -> int:
    return max(4, round(v / 4) * 4)


def sanitize_measure(events: list[Event], num: int, den: int) -> list[Event]:
    """Snap onsets and durations to something notatable. Binary values (multiples of 3, a 32nd)
    stay; a beat holding anything only the triplet grid explains (multiples of 4) goes wholly
    onto that grid, so its tuplet is complete; everything else moves to the nearest 32nd. Triplet
    notes are kept inside their beat, except quarter-note triplets laid over a half."""
    beat = beat_length(num, den)
    compound = is_compound(num, den)
    by_beat: dict[int, list[Event]] = {}
    for ev in events:
        if ev.pos < 0 or ev.dur <= 0: continue
        by_beat.setdefault(ev.pos // beat, []).append(ev)
    out = []
    for b, evs in by_beat.items():
        start = b * beat
        triplet = (not compound) and any(
            ((ev.pos - start) % 3 or ev.dur % 3) and (ev.pos - start) % 4 == 0 and ev.dur % 4 == 0 for ev in evs)
        for ev in evs:
            rel = ev.pos - start
            if triplet:
                rel = _snap4(rel) if rel else 0
                dur = _snap4(ev.dur)
                if rel % 3 or dur % 3:   # really on the triplet grid: keep it inside its beat
                    if not (rel % 16 == 0 and dur % 16 == 0 and start % (2 * beat) == 0):
                        dur = min(dur, beat - rel)
                        if dur < 4: continue
            else:
                rel = _snap3(rel) if rel else 0
                dur = _snap3(ev.dur)
            ev.pos, ev.dur = start + rel, dur
            out.append(ev)
    out.sort(key=lambda e: (e.pos, -e.top))
    return out


def clip_to_bar(events: list[Event], bar: int, next_bar: int | None) -> tuple[list[Event], list[Event]]:
    """Keep every event inside its bar. A note reaching past the barline is cut there and, when
    there is a following bar, continued in it as a tied note that starts on the downbeat; the
    continuation itself is clipped to that bar, so a note spills one bar at most. Returns
    (kept, spawned-into-next-bar)."""
    kept, spawned = [], []
    for ev in events:
        if ev.pos >= bar: continue
        if ev.end > bar:
            spill = ev.end - bar
            ev.dur = bar - ev.pos
            if next_bar is not None and spill >= 3:
                ev.tie_start = True
                spawned.append(Event(0, min(spill, next_bar),
                                     [NoteIn(n.midi, None, tie_stop=True) for n in ev.notes], uid=ev.uid))
        kept.append(ev)
    return kept, spawned


def merge_chords(events: list[Event]) -> list[Event]:
    """Notes sharing an onset and a duration are one chord."""
    slots: dict[tuple[int, int], Event] = {}
    for ev in events:
        k = (ev.pos, ev.dur)
        if k in slots:
            have = {n.midi for n in slots[k].notes}
            slots[k].notes.extend(n for n in ev.notes if n.midi not in have)
            slots[k].tie_start = slots[k].tie_start or ev.tie_start
        else:
            slots[k] = ev
    out = list(slots.values())
    for ev in out:
        ev.notes.sort(key=lambda n: n.midi)
    out.sort(key=lambda e: (e.pos, -e.top))
    return out


def assign_voices(events: list[Event]) -> tuple[list[list[Event]], set[int]]:
    """Lay one bar's events out over at most MAX_VOICES voices, top voice first.

    Onsets are walked in time order, highest pitch first. An event goes to the voice whose last
    note is nearest in pitch among the voices that are free at its onset (a free new voice counts
    as NEW_VOICE_COST away). When no voice is free, the sounding note nearest in pitch is cut
    short at the new onset -- or, at an identical onset, the new notes join that voice's chord --
    rather than a fifth voice opening. Voices that never overlap are then folded together, and
    the survivors are ordered top to bottom by mean pitch.

    Returns the voices and the uids of cut notes whose barline continuation must be dropped.
    """
    voices: list[list[Event]] = []
    cancelled: set[int] = set()
    for ev in events:
        p = ev.mean
        free = [i for i, v in enumerate(voices) if v[-1].end <= ev.pos]
        costs = [(abs(voices[i][-1].mean - p), i) for i in free]
        if len(voices) < MAX_VOICES:
            costs.append((NEW_VOICE_COST, len(voices)))
        if costs:
            _, i = min(costs)
            if i == len(voices): voices.append([])
            voices[i].append(ev)
            continue
        _, i = min((abs(v[-1].mean - p), i) for i, v in enumerate(voices))
        last = voices[i][-1]
        if last.pos == ev.pos:
            have = {n.midi for n in last.notes}
            last.notes.extend(n for n in ev.notes if n.midi not in have)
            last.notes.sort(key=lambda n: n.midi)
            if ev.tie_start: cancelled.add(ev.uid)
        else:
            last.dur = ev.pos - last.pos
            if last.tie_start:
                last.tie_start = False
                cancelled.add(last.uid)
            voices[i].append(ev)

    def overlap(a: list[Event], b: list[Event]) -> bool:
        return any(x.pos < y.end and y.pos < x.end for x in a for y in b)

    for i in range(len(voices)):
        for j in range(i + 1, len(voices)):
            if voices[i] and voices[j] and not overlap(voices[i], voices[j]):
                voices[i] = sorted(voices[i] + voices[j], key=lambda e: e.pos)
                voices[j] = []
    voices = [v for v in voices if v]
    voices.sort(key=lambda v: -sum(e.mean for e in v) / len(v))
    return voices, cancelled


# ---------------------------------------------------------------- rhythm

def _triplet_segments(o: int, d: int) -> list[tuple[int, int]]:
    segs = []
    while d > 0:
        for lvl in TRIPLET_DURATIONS:
            if lvl <= d and o % lvl == 0:
                break
        else:
            lvl = min(d, 4)
        segs.append((o, lvl))
        o, d = o + lvl, d - lvl
    return segs


def split_note(pos: int, dur: int, num: int, den: int) -> list[tuple[int, int]]:
    """The (pos, dur) pieces a note is written as, tied together, so that it never obscures a
    metric boundary stronger than itself. A note may cross a boundary only if the node starting
    there is no bigger than the note (its undotted value, for dotted notes) and the note starts
    on the half-of-its-own-length grid -- which admits the standard syncopations (eighth-quarter-
    eighth, quarter-half-quarter, eighth-dotted-quarter) and nothing looser. In compound meters a
    beat may only be crossed by a note that starts on a beat and lasts whole beats. Triplets are
    never split."""
    levels = metric_levels(num, den)
    beat, compound = beat_length(num, den), is_compound(num, den)
    segs: list[tuple[int, int]] = []

    def rec(o: int, d: int):
        if o % 3 or d % 3:
            segs.extend(_triplet_segments(o, d))
            return
        crossed = [(node_size(t, levels), t) for t in range(o + 3, o + d, 3)]
        if d in PLAIN_DURATIONS or d in DOTTED_DURATIONS:
            u = d if d in PLAIN_DURATIONS else d * 2 // 3
            ok = o % max(u // 2, 1) == 0
            for size, t in crossed:
                if compound and t % beat == 0 and not (o % beat == 0 and d % beat == 0):
                    ok = False
                elif size > u:
                    ok = False
            if ok:
                segs.append((o, d))
                return
        if not crossed:
            segs.append((o, d))
            return
        size, t = max(crossed, key=lambda st: (st[0], -st[1]))
        rec(o, t - o)
        rec(t, o + d - t)

    rec(pos, dur)
    return segs


def partition_rest(a: int, b: int, num: int, den: int) -> list[tuple[int, int]]:
    """The rests that fill [a, b): the largest aligned nodes of the metric tree first, so rests
    show the beat structure (no rest straddles a beat it does not start on, no dotted rests below
    the compound beat), with tuplet rests filling out a triplet beat."""
    levels = metric_levels(num, den)
    beat = beat_length(num, den)
    out = []
    while a < b:
        beat_start = a - a % beat
        beat_end = min(b, beat_start + beat)
        if a % 3 or beat_end % 3:
            while a < beat_end:
                lvl = next((l for l in TRIPLET_DURATIONS if (a - beat_start) % l == 0 and a + l <= beat_end), None)
                if lvl is None: lvl = beat_end - a
                out.append((a, lvl))
                a += lvl
            continue
        for lvl in levels:
            if a % lvl == 0 and a + lvl <= b:
                out.append((a, lvl))
                a += lvl
                break
        else:
            out.append((a, b - a))
            a = b
    return out


def grid_duration(units: int) -> duration21.Duration:
    return duration21.Duration(Fraction(units, SUBDIVISIONS_PER_QUARTER))


# ---------------------------------------------------------------- spelling

_BLACK = {1: ('C#', 'D-'), 3: ('D#', 'E-'), 6: ('F#', 'G-'), 8: ('G#', 'A-'), 10: ('A#', 'B-')}
# how a chromatic note is usually spelled, by its interval above the tonic: the raised tones are
# secondary leading tones (to ii, V and vi), the lowered ones the borrowed bIII and bVII
_CHROMATIC_SHARP = {1, 6, 8}
_scale_cache: dict[int, dict[int, str]] = {}


def _scale_names(sharps: int) -> dict[int, str]:
    if sharps not in _scale_cache:
        _scale_cache[sharps] = {p.pitchClass: p.name for p in key21.KeySignature(sharps).getScale('major').pitches}
    return _scale_cache[sharps]


def _with_octave(name: str, midi: int) -> pitch21.Pitch:
    p = pitch21.Pitch(name)
    p.octave = midi // 12 - 1
    while p.midi < midi: p.octave += 1
    while p.midi > midi: p.octave -= 1
    return p


def _simple(name: str) -> bool:
    """A plain name: at most one accidental, and not one of the white-key sharps/flats."""
    return name not in ('B#', 'E#', 'C-', 'F-') and not (name.endswith('##') or name.endswith('--'))


def spell_pitch(midi: int, sharps: int | None = 0, chord_pitches=None,
                prev_midi: int | None = None, next_midi: int | None = None,
                prior: dict[int, str] | None = None) -> pitch21.Pitch:
    """The spelling of a midi pitch in context. In order: the key's own diatonic name; the name
    the chord in effect gives it; the name the same pitch class already has in this bar (`prior`,
    pitch class -> name); a chromatic note approaching its neighbour by semitone, spelled towards
    it; else the usual spelling of that chromatic degree (raised leading tones to ii, V and vi,
    lowered bIII and bVII). Never a double accidental, never B#/E#/Cb/Fb unless the key itself
    has them."""
    sharps = sharps or 0
    pc = midi % 12
    names = _scale_names(sharps)
    if pc in names:
        return _with_octave(names[pc], midi)
    if pc not in _BLACK:
        return _with_octave(pitch21.Pitch(midi=midi).step, midi)   # a white key outside the key is its natural
    sharp_name, flat_name = _BLACK[pc]
    if chord_pitches:
        for cp in chord_pitches:
            if cp.pitchClass == pc and _simple(cp.name):
                return _with_octave(cp.name, midi)
    if prior and pc in prior:
        return _with_octave(prior[pc], midi)
    if next_midi is not None and next_midi - midi == 1:
        return _with_octave(sharp_name, midi)
    if next_midi is not None and midi - next_midi == 1:
        return _with_octave(flat_name, midi)
    if prev_midi is not None and prev_midi - midi == 1 and next_midi is None:
        return _with_octave(flat_name, midi)
    tonic = (7 * sharps) % 12
    return _with_octave(sharp_name if (pc - tonic) % 12 in _CHROMATIC_SHARP else flat_name, midi)


class ChordContext:
    """The chord symbol in effect at any (bar, pos), from rows placed in score order."""

    def __init__(self):
        self._keys: list[tuple[int, int]] = []
        self._pitches: list[tuple] = []

    def add(self, m: int, pos: int, symbol: harmony21.ChordSymbol):
        try:
            pitches = tuple(symbol.pitches)
        except Exception:
            return
        i = bisect.bisect_right(self._keys, (m, pos))
        self._keys.insert(i, (m, pos))
        self._pitches.insert(i, pitches)

    def at(self, m: int, pos: int):
        i = bisect.bisect_right(self._keys, (m, pos))
        return self._pitches[i - 1] if i else None


# ---------------------------------------------------------------- one bar of one staff

def _nearest(midi: int, ev: Event | None) -> int | None:
    if ev is None: return None
    return min((n.midi for n in ev.notes), key=lambda x: abs(x - midi))


def voice_elements(events: list[Event], num: int, den: int, m: int, sharps: int | None,
                   chords: ChordContext | None, stem: str | None,
                   label_text=None) -> list[note21.GeneralNote]:
    """The notes, chords and rests of one voice of one bar, with offsets set: events split and
    tied by split_note, gaps filled by partition_rest, pitches spelled in context. An empty voice
    is a single full-measure rest."""
    bar = bar_length(num, den)
    out: list[note21.GeneralNote] = []
    if not events:
        r = note21.Rest(quarterLength=Fraction(bar, SUBDIVISIONS_PER_QUARTER))
        r.fullMeasure = True
        r.offset = 0
        return [r]

    def rests(a: int, b: int):
        for o, d in partition_rest(a, b, num, den):
            r = note21.Rest()
            r.duration = grid_duration(d)
            r.offset = Fraction(o, SUBDIVISIONS_PER_QUARTER)
            out.append(r)

    cursor = 0
    prior: dict[int, str] = {}   # pitch class -> how it was spelled earlier in this bar
    for idx, ev in enumerate(events):
        if ev.pos > cursor: rests(cursor, ev.pos)
        prev_ev = events[idx - 1] if idx else None
        next_ev = events[idx + 1] if idx + 1 < len(events) else None
        chord_pitches = chords.at(m, ev.pos) if chords else None
        spelled = [spell_pitch(n.midi, sharps, chord_pitches, _nearest(n.midi, prev_ev), _nearest(n.midi, next_ev), prior)
                   for n in ev.notes]
        for p in spelled: prior.setdefault(p.pitchClass, p.name)
        segs = split_note(ev.pos, ev.dur, num, den)
        for k, (o, d) in enumerate(segs):
            if len(spelled) == 1:
                gn: note21.NotRest = note21.Note(spelled[0])
                heads = [gn]
            else:
                gn = chord21.Chord(spelled)
                heads = list(gn.notes)
            gn.duration = grid_duration(d)
            gn.offset = Fraction(o, SUBDIVISIONS_PER_QUARTER)
            if stem: gn.stemDirection = stem
            for head, n in zip(heads, ev.notes):
                if n.notehead: head.notehead = n.notehead
            next_linked = k < len(segs) - 1 or ev.tie_start
            for head, n in zip(heads, ev.notes):
                prev_linked = k > 0 or n.tie_stop
                if prev_linked and next_linked: head.tie = tie21.Tie('continue')
                elif next_linked: head.tie = tie21.Tie('start')
                elif prev_linked: head.tie = tie21.Tie('stop')
            if k == 0 and label_text:
                text = label_text([n.label for n in ev.notes])
                if text: gn.lyric = text
            out.append(gn)
        cursor = ev.end
    if cursor < bar: rests(cursor, bar)
    return out


def fill_measure(measure: stream21.Measure, voices: list[list[Event]], num: int, den: int, m: int,
                 sharps: int | None, chords: ChordContext | None, label_text=None) -> None:
    """Write one staff's bar into `measure`: directly when there is one voice, else as Voice
    objects numbered from 1 with stems fixed by voice."""
    if len(voices) <= 1:
        for el in voice_elements(voices[0] if voices else [], num, den, m, sharps, chords, None, label_text):
            measure.insert(el.offset, el)
        return
    for i, events in enumerate(voices):
        v = stream21.Voice(id=str(i + 1))
        for el in voice_elements(events, num, den, m, sharps, chords, STEM_DIRECTIONS[i], label_text):
            v.insert(el.offset, el)
        measure.insert(0, v)


def layout_staff(events_by_bar: dict[int, list[Event]], meters: dict[int, tuple[int, int]],
                 first: int, last: int) -> dict[int, list[list[Event]]]:
    """Voices for every bar of one staff, bars `first`..`last`, from raw events keyed by bar.
    Bars are walked in order so that a note cut at a barline lands in the next bar before that
    bar is laid out; `meters[m]` is the (num, den) in effect at bar m."""
    spawned: dict[int, list[Event]] = {}
    out: dict[int, list[list[Event]]] = {}
    for m in range(first, last + 1):
        num, den = meters[m]
        bar = bar_length(num, den)
        next_bar = bar_length(*meters[m + 1]) if m < last else None
        events = sanitize_measure([*events_by_bar.get(m, []), *spawned.pop(m, [])], num, den)
        events, into_next = clip_to_bar(events, bar, next_bar)
        events = merge_chords(events)
        voices, cancelled = assign_voices(events)
        into_next = [ev for ev in into_next if ev.uid not in cancelled]
        if into_next: spawned[m + 1] = into_next
        out[m] = voices
    return out


# ---------------------------------------------------------------- whole-score passes

def sounding(container) -> list[note21.GeneralNote]:
    """The notes and chords of a container, in offset order. Chord symbols are excluded: music21
    models one as a Chord, so every pass that walks a bar's notes would otherwise read the
    harmony written above the staff as if it were played on it -- filling no rest under it,
    printing its accidentals, and respelling it."""
    return [el for el in container.getElementsByClass(note21.NotRest)
            if not isinstance(el, harmony21.Harmony)]


def _grid(offset) -> int | None:
    """`offset` (in quarters) as whole grid units, or None when it does not land on the grid."""
    units = Fraction(offset) * SUBDIVISIONS_PER_QUARTER
    return int(units) if units.denominator == 1 else None


def _holes(container, span: float) -> list[tuple[float, float]]:
    """The stretches of [0, span) that nothing in `container` sounds through."""
    spans = sorted((float(el.offset), float(el.offset) + float(el.duration.quarterLength))
                   for el in [*sounding(container), *container.getElementsByClass(note21.Rest)])
    holes, cursor = [], 0.0
    for start, end in spans:
        if start > cursor: holes.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < span: holes.append((cursor, span))
    return holes


def fill_rests(score: stream21.Score) -> None:
    """Make every bar read as a full bar. A bar (or voice) with nothing played in it becomes a
    single whole -- full-measure -- rest; a bar with holes in it has them filled with rests
    grouped by the metric tree; and a rest that arrived marked invisible is shown.

    music21's own exporter fills a hole with an invisible rest, which engraves as a blank gap,
    and a bar left blank on the page is a bar the reader cannot count. A voice that plays nothing
    while another voice in the same bar does is dropped instead: the bar is not silent, so a rest
    of its own would only collide with the notes beside it.
    """
    for part in score.parts:
        meter_now = (4, 4)
        for m in part.getElementsByClass(stream21.Measure):
            ts = m.timeSignature or m.getContextByClass(meter21.TimeSignature)
            if ts is not None: meter_now = (ts.numerator, ts.denominator)
            num, den = meter_now
            # a pickup, or any deliberately short bar, is only as long as it says it is
            span = float(m.barDuration.quarterLength) - float(m.paddingLeft) - float(m.paddingRight)
            voices = list(m.voices)
            played = [v for v in voices if sounding(v)]
            if voices and not played:
                # nothing is played anywhere in the bar: it is one silent bar, not several
                for v in voices: m.remove(v)
                voices = []
            for c in (voices or [m]):
                for rest in c.getElementsByClass(note21.Rest):
                    rest.style.hideObjectOnPrint = False
                if sounding(c): continue
                if voices:
                    m.remove(c)      # another voice carries this bar; a second rest would only clash
                    continue
                for rest in list(c.getElementsByClass(note21.Rest)):
                    c.remove(rest)
                whole = note21.Rest(quarterLength=span if span > 0 else m.barDuration.quarterLength)
                whole.fullMeasure = True
                c.insert(0, whole)
            for c in (list(m.voices) or [m]):
                if not sounding(c): continue
                for a, b in _holes(c, span):
                    ga, gb = _grid(a), _grid(b)
                    pieces = (partition_rest(ga, gb, num, den)
                              if ga is not None and gb is not None and gb > ga else None)
                    for o, d in pieces if pieces is not None else [(None, None)]:
                        rest = note21.Rest()
                        if o is None:
                            rest.duration = duration21.Duration(Fraction(b - a).limit_denominator(1024))
                            c.insert(Fraction(a).limit_denominator(1024), rest)
                        else:
                            rest.duration = grid_duration(d)
                            c.insert(Fraction(o, SUBDIVISIONS_PER_QUARTER), rest)


def _voice_index(el, measure: stream21.Measure) -> int:
    site = el.activeSite
    if site is None or site is measure: return 0
    for i, v in enumerate(measure.voices):
        if v is site: return i
    return 0


def make_accidentals(part: stream21.Part) -> None:
    """Decide which accidentals print, bar by bar, walking notes in time order across voices
    (music21's own pass walks voice by voice). An accidental prints once per bar unless a
    contradicting pitch intervenes; a tied-over note repeats nothing; a note contradicting the
    previous bar gets a cautionary; the key signature's own alterations print nothing."""
    measures = list(part.getElementsByClass(stream21.Measure))
    ks = key21.KeySignature(0)
    ks_names: list[str] = []
    past_measure: list[pitch21.Pitch] = []
    for i, m in enumerate(measures):
        if m.keySignature is not None:
            if i > 0 and ks_names:
                past_measure = [p for p in past_measure if p.name not in ks_names]
            ks = m.keySignature
            ks_names = [p.name for p in ks.getScale().pitches]
        altered = ks.alteredPitches
        past: list[pitch21.Pitch] = []
        elements = sorted([el for v in ([*m.voices] or [m]) for el in sounding(v)],
                          key=lambda el: (el.getOffsetInHierarchy(m), _voice_index(el, m)))
        for el in elements:
            if isinstance(el, chord21.Chord):
                heads = list(el.notes)
            elif isinstance(el, note21.Note):
                heads = [el]
            else:
                continue
            for head in heads:
                p = head.pitch
                if head.notehead == 'slash':   # a placeholder, not a pitch: never an accidental
                    p.accidental = None
                    continue
                tied_over = head.tie is not None and head.tie.type in ('stop', 'continue')
                others = [o for o in el.pitches if o is not p] if len(heads) > 1 else None
                p.updateAccidentalDisplay(
                    pitchPast=past, pitchPastMeasure=past_measure, otherSimultaneousPitches=others,
                    alteredPitches=altered, cautionaryPitchClass=True, cautionaryAll=False,
                    overrideStatus=True, cautionaryNotImmediateRepeat=False, lastNoteWasTied=tied_over)
            past.extend(head.pitch for head in heads)
        past_measure = past
        m.streamStatus.accidentals = True


_STATE_CLASSES = (clef21.Clef, key21.KeySignature, meter21.TimeSignature, tempo21.MetronomeMark)


def _state_of(el) -> tuple[str, object]:
    """Which piece of running state an element sets, and the value it sets it to. A tempo mark
    with no number (a bare "Andante") comes back valueless: it is never a restatement."""
    if isinstance(el, clef21.Clef): return 'clef', (el.sign, el.line, el.octaveChange)
    if isinstance(el, key21.KeySignature): return 'key', el.sharps
    if isinstance(el, meter21.TimeSignature): return 'time', el.ratioString
    return 'tempo', el.number


def _state_marks(measure: stream21.Measure) -> list:
    """The clefs, key and time signatures and tempo marks of a bar, in the order they take effect."""
    return sorted((el for el in measure.recurse().getElementsByClass(_STATE_CLASSES) if _state_of(el)[1] is not None),
                  key=lambda el: el.getOffsetInHierarchy(measure))


def drop_restated(score: stream21.Score) -> None:
    """Remove every clef, key signature, time signature and tempo mark that only restates what is
    already in effect on its staff. These are changes of state, not content, but they are stored
    on the bar they happen in -- so a bar that is copied, generated or pasted somewhere else
    brings its own along, and without this the score opens again wherever one lands."""
    for part in score.parts:
        state: dict[str, object] = {}
        for m in part.getElementsByClass(stream21.Measure):
            for el in _state_marks(m):
                kind, value = _state_of(el)
                if state.get(kind) == value:
                    m.remove(el, recurse=True)
                else:
                    state[kind] = value


def carry_state(measures: list[stream21.Measure], index: int) -> None:
    """Write the clef, key, time signature and tempo in effect as `measures[index]` begins onto
    that bar itself, wherever it does not open by setting its own. For the bar that follows a
    block about to be removed or replaced: what it inherited from the bars before it is about to
    change under it, and this is what keeps it reading the way it did. Anything written here
    that turns out to restate the new bars before it is dropped again by drop_restated."""
    if not 0 < index < len(measures): return
    state: dict[str, object] = {}
    for m in measures[:index]:
        for el in _state_marks(m):
            state[_state_of(el)[0]] = el
    target = measures[index]
    opening = {_state_of(el)[0] for el in _state_marks(target) if el.getOffsetInHierarchy(target) == 0}
    for kind, el in state.items():
        if kind not in opening:
            target.insert(0, copy.deepcopy(el))


def engrave(score: stream21.Score) -> None:
    """Beams, tuplet brackets and accidental display for every part, in place. Measures must
    already be inside their parts (beaming reads the time signature from context)."""
    fill_rests(score)
    for part in score.parts:
        make_accidentals(part)
        for m in part.getElementsByClass(stream21.Measure):
            try:
                m.makeBeams(inPlace=True, setStemDirections=False)
            except Exception:   # a bar with no meter in reach simply goes unbeamed
                log.debug('no beams for measure %s', m.number, exc_info=True)
            for s in [m, *m.voices]:
                makeNotation.makeTupletBrackets(s, inPlace=True)


def export_xml(score: stream21.Score) -> str:
    """MusicXML for a score whose notation is already made. The exporter is told not to make
    notation itself: left to it, it fills gaps with invisible rests and re-derives accidentals
    with music21's repeat-the-cautionary rule, undoing the passes above."""
    exporter = music21.musicxml.m21ToXml.GeneralObjectExporter(score)
    exporter.makeNotation = False
    return exporter.parse().decode('utf-8')


def finish(score: stream21.Score) -> str:
    """engrave + export, for producers holding a complete score."""
    drop_restated(score)
    engrave(score)
    return export_xml(score)


def respell(score: stream21.Score, from_measure: int | None = None) -> None:
    """Respell every note in place from its own bar's key signature, chord symbols and neighbours,
    for scores edited after they were engraved (a key change, say). Ties are kept by pitch."""
    for part in score.parts:
        sharps = 0
        for m in part.getElementsByClass(stream21.Measure):
            if m.keySignature is not None: sharps = m.keySignature.sharps
            if from_measure is not None and (m.number or 0) < from_measure: continue
            symbols = sorted(m.recurse().getElementsByClass(harmony21.ChordSymbol),
                             key=lambda c: c.getOffsetInHierarchy(m))
            containers = [m, *m.voices] if m.voices else [m]
            for c in containers:
                notes = sounding(c)
                prior: dict[int, str] = {}
                for idx, el in enumerate(notes):
                    prev_ev = notes[idx - 1] if idx else None
                    next_ev = notes[idx + 1] if idx + 1 < len(notes) else None
                    off = el.getOffsetInHierarchy(m)
                    chord_pitches = None
                    for cs in symbols:
                        if cs.getOffsetInHierarchy(m) <= off:
                            try: chord_pitches = tuple(cs.pitches)
                            except Exception: pass
                    heads = [el] if isinstance(el, note21.Note) else list(el.notes)
                    for head in heads:
                        midi = head.pitch.midi
                        near = lambda ev, midi=midi: None if ev is None else min((p.midi for p in ev.pitches), key=lambda x: abs(x - midi))
                        head.pitch = spell_pitch(midi, sharps, chord_pitches, near(prev_ev), near(next_ev), prior)
                        prior.setdefault(head.pitch.pitchClass, head.pitch.name)


def finish_edited(score: stream21.Score, respell_from: int | None = None) -> str:
    """Export for a score that was parsed from MusicXML and edited: its rhythm and beams are kept
    as they came, blank bars are filled, accidentals are decided afresh (music21's reader leaves
    them undecided, and its exporter would decide them with the repeat-the-cautionary rule), and
    with `respell_from` every note from that bar on is respelled for a changed key."""
    drop_restated(score)
    fill_rests(score)
    if respell_from is not None:
        respell(score, respell_from)
    for part in score.parts:
        make_accidentals(part)
    return export_xml(score)
