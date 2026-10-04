"""The engraving rules, checked on the MusicXML they produce from hand-built rows."""

import re
import xml.etree.ElementTree as ET

import pytest

import notation
from notation import Event, NoteIn, partition_rest, split_note, spell_pitch
from tokenizer import tokens_to_score

HEADER = [('SOG', '', '', '', ''), ('INSTR_piano', '', '', '', 'VOICE_0'), ('SOS', '', '', '', '')]


def note(m, pos, dur, midi, voice=0):
    return (f'M_{m}', f'POS_{pos}', f'DUR_{dur}', f'PITCH_{midi}', f'VOICE_{voice}')


def render(rows, key=0, time=(4, 4)):
    head = [('M_0', 'POS_0', f'KEY_{key}', '', ''), ('M_0', 'POS_0', f'NUM_{time[0]}', f'DEN_{time[1]}', '')]
    return tokens_to_score([*HEADER, *head, *rows, ('EOS', '', '', '', '')])


def measures(xml):
    root = ET.fromstring(xml)
    return [m for part in root.findall('part') for m in part.findall('measure')]


def notes_of(measure):
    """(step+accidental name, duration, type, tie types, voice, is_rest) per note element."""
    out = []
    for n in measure.findall('note'):
        rest = n.find('rest') is not None
        step = n.findtext('pitch/step', '') + {'1': '#', '-1': 'b', '0': '', '': ''}[n.findtext('pitch/alter', '')]
        out.append(dict(name=step, dur=int(n.findtext('duration')), type=n.findtext('type'),
                        ties=sorted(t.get('type') for t in n.findall('tie')), voice=n.findtext('voice'),
                        rest=rest, accidental=n.findtext('accidental'), stem=n.findtext('stem'),
                        octave=n.findtext('pitch/octave'),
                        full=(n.find('rest') is not None and n.find('rest').get('measure') == 'yes')))
    return out


def bar_sum(measure, divisions):
    """Duration of each voice of a measure in quarters."""
    sums = {}
    for n in measure.findall('note'):
        if n.find('chord') is not None: continue
        sums[n.findtext('voice', '1')] = sums.get(n.findtext('voice', '1'), 0) + int(n.findtext('duration'))
    return {v: s / divisions for v, s in sums.items()}


# ---------------------------------------------------------------- rhythm rules in isolation

@pytest.mark.parametrize('pos,dur,expected', [
    (0, 96, [(0, 96)]),               # whole note
    (24, 48, [(24, 48)]),             # quarter - half - quarter
    (12, 24, [(12, 24)]),             # eighth - quarter - eighth
    (36, 24, [(36, 12), (48, 12)]),   # a quarter on 2-and crossing the middle: two tied eighths
    (60, 24, [(60, 24)]),             # a quarter on 3-and only crosses beat 4: fine
    (12, 48, [(12, 36), (48, 12)]),   # a half on 1-and: dotted quarter tied to an eighth
    (12, 36, [(12, 36)]),             # eighth - dotted quarter
    (36, 36, [(36, 12), (48, 24)]),   # dotted quarter on 2-and crossing the middle
    (0, 15, [(0, 12), (12, 3)]),      # not a note value: split at the strongest boundary inside
])
def test_split_note_simple_meter(pos, dur, expected):
    assert split_note(pos, dur, 4, 4) == expected


@pytest.mark.parametrize('pos,dur,expected', [
    (12, 24, [(12, 24)]),             # eighth - quarter within a beat of 6/8
    (24, 24, [(24, 12), (36, 12)]),   # a quarter crossing the compound beat
    (0, 48, [(0, 36), (36, 12)]),     # a half in 6/8: dotted quarter tied to eighth
    (0, 72, [(0, 72)]),               # the whole bar
])
def test_split_note_compound_meter(pos, dur, expected):
    assert split_note(pos, dur, 6, 8) == expected


def test_split_note_leaves_triplets_alone():
    assert split_note(8, 8, 4, 4) == [(8, 8)]
    assert split_note(16, 16, 4, 4) == [(16, 16)]


@pytest.mark.parametrize('a,b,expected', [
    (0, 96, [(0, 96)]),
    (0, 48, [(0, 48)]),               # half rest on beat 1
    (48, 96, [(48, 48)]),             # half rest on beat 3
    (24, 72, [(24, 24), (48, 24)]),   # never a half rest across the middle
    (24, 42, [(24, 12), (36, 6)]),    # eighth + sixteenth, no dotted rest
    (6, 24, [(6, 6), (12, 12)]),
])
def test_partition_rest_4_4(a, b, expected):
    assert partition_rest(a, b, 4, 4) == expected


def test_partition_rest_6_8_uses_dotted_quarter_on_the_beat():
    assert partition_rest(0, 36, 6, 8) == [(0, 36)]
    assert partition_rest(12, 48, 6, 8) == [(12, 12), (24, 12), (36, 12)]


def test_partition_rest_fills_a_triplet_beat_with_tuplet_rests():
    assert partition_rest(8, 24, 4, 4) == [(8, 8), (16, 8)]


# ---------------------------------------------------------------- spelling in isolation

def test_spelling_follows_the_key():
    assert spell_pitch(70, -2).name == 'B-'          # Bb in Bb major, never A#
    assert spell_pitch(70, 0).name == 'B-'           # bVII of C: the usual flat
    assert spell_pitch(66, 0).name == 'F#'           # raised 4th of C: the usual sharp
    assert spell_pitch(63, 0).name == 'E-'           # bIII of C
    assert spell_pitch(65, 7).name == 'E#'           # diatonic in C# major
    assert spell_pitch(71, -3).name == 'B'           # leading tone of C minor, not Cb
    assert spell_pitch(61, -1).name == 'C#'          # leading tone of D minor, not Db
    assert spell_pitch(68, 0).name == 'G#'           # leading tone of A minor


def test_spelling_follows_chord_then_melody_then_key():
    import music21
    e_major = tuple(music21.harmony.ChordSymbol('E').pitches)
    assert spell_pitch(68, 0, e_major).name == 'G#'
    ab_major = tuple(music21.harmony.ChordSymbol('A-').pitches)
    assert spell_pitch(68, 0, ab_major).name == 'A-'
    assert spell_pitch(63, 0, None, None, 64).name == 'D#'   # rising to E
    assert spell_pitch(63, 0, None, None, 62).name == 'E-'   # falling to D
    assert spell_pitch(63, 2, None, None, None).name == 'D#' # sharp key preference
    assert spell_pitch(63, -4, None, None, None).name == 'E-'


def test_spelling_never_uses_odd_white_key_names_outside_the_key():
    assert spell_pitch(64, -2).name == 'E'   # E natural in Bb major, not Fb
    assert spell_pitch(65, 3).name == 'F'    # F natural in A major, not E#


# ---------------------------------------------------------------- whole renders

def test_every_bar_is_complete_with_visible_rests():
    xml = render([note(0, 0, 24, 60), note(0, 48, 12, 62), note(1, 36, 24, 64)])
    assert 'print-object="no"' not in xml
    divisions = int(re.search(r'<divisions>(\d+)', xml).group(1))
    for m in measures(xml):
        assert all(s == 4 for s in bar_sum(m, divisions).values()), ET.tostring(m)
    m0 = notes_of(measures(xml)[0])
    assert [n['rest'] for n in m0] == [False, True, False, True, True]
    assert [n['type'] for n in m0] == ['quarter', 'quarter', 'eighth', 'eighth', 'quarter']


def test_gap_at_the_start_and_syncopation_across_the_middle():
    xml = render([note(0, 36, 24, 60)])
    m0 = notes_of(measures(xml)[0])
    assert [(n['rest'], n['type'], n['ties']) for n in m0] == [
        (True, 'quarter', []), (True, 'eighth', []), (False, 'eighth', ['start']), (False, 'eighth', ['stop']),
        (True, 'eighth', []), (True, 'quarter', [])]


def test_a_spelling_is_kept_for_the_rest_of_the_bar():
    xml = render([note(0, 0, 24, 66), note(0, 24, 24, 67), note(0, 48, 24, 66), note(0, 72, 24, 65)])
    assert [n['name'] for n in notes_of(measures(xml)[0])] == ['F#', 'G', 'F#', 'F']


def test_empty_bar_is_a_measure_rest():
    xml = render([note(0, 0, 96, 60), note(2, 0, 96, 60)])
    m1 = notes_of(measures(xml)[1])
    assert len(m1) == 1 and m1[0]['full']


def test_a_silent_staff_gets_a_whole_rest_in_every_bar():
    two = [('SOG', '', '', '', ''), ('INSTR_piano', '', '', '', 'VOICE_0'),
           ('INSTR_piano', '', '', '', 'VOICE_1'), ('SOS', '', '', '', ''),
           ('M_0', 'POS_0', 'NUM_4', 'DEN_4', '')]
    xml = tokens_to_score([*two, note(0, 0, 96, 60), note(2, 0, 96, 62), ('EOS', '', '', '', '')])
    parts = ET.fromstring(xml).findall('part')
    assert len(parts) == 2 and all(len(p.findall('measure')) == 3 for p in parts)
    for part in parts:
        for m in part.findall('measure'):
            written = notes_of(m)
            assert written, ET.tostring(m)
            if all(n['rest'] for n in written):
                assert len(written) == 1 and written[0]['full'], ET.tostring(m)


def test_a_bar_holding_only_a_chord_symbol_is_still_a_whole_rest():
    rows = [('M_0', 'POS_0', 'NUM_4', 'DEN_4', ''), note(0, 0, 96, 60),
            ('M_1', 'POS_0', 'DEG_7', 'ROOT_7', 'QUAL_'), note(2, 0, 96, 60)]
    xml = tokens_to_score([*HEADER, *rows, ('EOS', '', '', '', '')])
    assert '<harmony' in xml
    m1 = notes_of(measures(xml)[1])
    assert len(m1) == 1 and m1[0]['full'] and m1[0]['rest']


def test_a_chord_symbol_does_not_print_or_suppress_accidentals():
    # an F#7 symbol over a bar whose notes are F naturals: the harmony is read, not played
    rows = [('M_0', 'POS_0', 'NUM_4', 'DEN_4', ''), ('M_0', 'POS_0', 'DEG_6', 'ROOT_6', 'QUAL_7'),
            note(0, 0, 48, 65), note(0, 48, 48, 65)]
    xml = tokens_to_score([*HEADER, *rows, ('EOS', '', '', '', '')])
    m0 = notes_of(measures(xml)[0])
    assert [n['name'] for n in m0] == ['F', 'F']
    assert [n['accidental'] for n in m0] == [None, None]


def test_a_tempo_is_written_once_above_the_top_staff():
    # two staves, and the same spot named twice (a generated header over the mark it replaces)
    two = [HEADER[0], HEADER[1], ('INSTR_piano', '', '', '', 'VOICE_1'), HEADER[2]]
    rows = [('M_0', 'POS_0', 'NUM_4', 'DEN_4', ''), ('M_0', 'POS_0', 'TEMPO_6', '', ''),
            ('M_0', 'POS_0', 'TEMPO_8', '', ''), note(0, 0, 96, 60), note(0, 0, 96, 48, voice=1)]
    root = ET.fromstring(tokens_to_score([*two, *rows, ('EOS', '', '', '', '')]))
    per_part = [len(part.findall('.//metronome')) for part in root.findall('part')]
    assert per_part == [1, 0]
    assert root.findtext('.//metronome/per-minute') == '96'


def test_an_imported_score_with_blank_and_gapped_bars_is_completed():
    import music21 as m21
    from notation import finish_edited
    part = m21.stream.Part(); part.insert(0, m21.instrument.Piano())
    m1 = m21.stream.Measure(number=1); m1.timeSignature = m21.meter.TimeSignature('3/4')
    m1.insert(0.0, m21.note.Note('C4', quarterLength=1))   # a hole from beat 2 to the barline
    part.append(m1)
    part.append(m21.stream.Measure(number=2))              # blank
    m3 = m21.stream.Measure(number=3)
    hidden = m21.note.Rest(quarterLength=3)                # a rest that came in invisible
    hidden.style.hideObjectOnPrint = True
    m3.insert(0.0, hidden)
    part.append(m3)
    score = m21.stream.Score(); score.insert(0, part)

    xml = finish_edited(score)
    assert 'print-object="no"' not in xml
    divisions = int(re.search(r'<divisions>(\d+)', xml).group(1))
    ms = measures(xml)
    assert len(ms) == 3
    for m in ms:
        assert all(s == 3 for s in bar_sum(m, divisions).values()), ET.tostring(m)
    # beats 2 and 3 of a 3/4 bar rest as one rest each: the meter has no half-bar level for a
    # half rest to sit on
    assert [(n['rest'], n['type']) for n in notes_of(ms[0])] == [
        (False, 'quarter'), (True, 'quarter'), (True, 'quarter')]
    for m in ms[1:]:
        written = notes_of(m)
        assert len(written) == 1 and written[0]['full'] and written[0]['rest']


def test_a_pickup_bar_is_not_padded_out_to_a_full_bar():
    import music21 as m21
    from notation import finish_edited
    part = m21.stream.Part(); part.insert(0, m21.instrument.Piano())
    pickup = m21.stream.Measure(number=0); pickup.timeSignature = m21.meter.TimeSignature('4/4')
    pickup.paddingLeft = 3.0
    pickup.insert(0.0, m21.note.Note('G4', quarterLength=1))
    part.append(pickup)
    m1 = m21.stream.Measure(number=1); m1.insert(0.0, m21.note.Note('C4', quarterLength=4))
    part.append(m1)
    score = m21.stream.Score(); score.insert(0, part)

    xml = finish_edited(score)
    divisions = int(re.search(r'<divisions>(\d+)', xml).group(1))
    assert bar_sum(measures(xml)[0], divisions) == {'1': 1.0}


def test_a_silent_voice_beside_a_playing_one_is_dropped_not_rested():
    import music21 as m21
    from notation import finish_edited
    part = m21.stream.Part(); part.insert(0, m21.instrument.Piano())
    m1 = m21.stream.Measure(number=1); m1.timeSignature = m21.meter.TimeSignature('4/4')
    v1 = m21.stream.Voice(id='1'); v1.insert(0.0, m21.note.Note('C5', quarterLength=4))
    v2 = m21.stream.Voice(id='2'); v2.insert(0.0, m21.note.Rest(quarterLength=4))
    m1.insert(0, v1); m1.insert(0, v2)
    part.append(m1)
    score = m21.stream.Score(); score.insert(0, part)

    written = notes_of(measures(finish_edited(score))[0])
    assert len(written) == 1 and not written[0]['rest']


def test_note_past_the_barline_is_tied_into_the_next_bar_only():
    xml = render([note(0, 72, 48, 60), note(1, 24, 24, 62)])
    m0, m1 = measures(xml)[:2]
    assert notes_of(m0)[-1]['ties'] == ['start']
    first = notes_of(m1)[0]
    assert first['ties'] == ['stop'] and first['dur'] == notes_of(m0)[-1]['dur']
    xml = render([note(0, 0, 9611280 // 420, 60)])   # the 953-quarter note from a real render
    assert len(measures(xml)) == 2
    assert all(n['full'] or n['name'] == 'C' for m in measures(xml) for n in notes_of(m))
    assert notes_of(measures(xml)[1])[0]['ties'] == ['stop']


def test_accidental_prints_once_per_bar_and_naturals_are_cautionary():
    xml = render([note(0, 0, 24, 66), note(0, 24, 24, 67), note(0, 48, 24, 66), note(0, 72, 24, 65),
                  note(1, 0, 24, 65)])
    m0 = notes_of(measures(xml)[0])
    assert [n['accidental'] for n in m0] == ['sharp', None, None, 'natural']
    assert notes_of(measures(xml)[1])[0]['accidental'] is None   # the bar before ended on F natural
    xml = render([note(0, 0, 24, 66), note(0, 24, 24, 67), note(0, 48, 24, 66), note(0, 72, 24, 69),
                  note(1, 0, 24, 65)])
    assert notes_of(measures(xml)[1])[0]['accidental'] == 'natural'   # F after F# in the bar before


def test_tied_over_accidental_is_not_repeated():
    xml = render([note(0, 72, 48, 66)])
    m0, m1 = measures(xml)[:2]
    assert notes_of(m0)[-1]['accidental'] == 'sharp'
    assert notes_of(m1)[0]['accidental'] is None


def test_key_signature_alterations_print_nothing():
    xml = render([note(0, 0, 24, 70), note(0, 24, 24, 63), note(0, 48, 24, 58), note(0, 72, 24, 69)], key=10)
    assert '<fifths>-2</fifths>' in xml
    m0 = notes_of(measures(xml)[0])
    assert [n['name'] for n in m0] == ['Bb', 'Eb', 'Bb', 'A']
    assert [n['accidental'] for n in m0] == [None, None, None, None]


def test_accidentals_are_tracked_in_time_order_across_voices():
    # F#4 on beat 1 in the lower voice, F#4 on beat 3 in the upper: the lower one prints, the upper doesn't
    xml = render([note(0, 0, 48, 84), note(0, 48, 48, 66),
                  note(0, 0, 24, 66), note(0, 24, 24, 67), note(0, 48, 24, 69), note(0, 72, 24, 71)])
    m0 = notes_of(measures(xml)[0])
    printed = {n['voice']: n['accidental'] for n in m0 if n['name'] == 'F#'}
    assert printed == {'1': None, '2': 'sharp'}


def test_overlapping_notes_become_voices_with_stems_by_voice():
    xml = render([note(0, 0, 96, 72), note(0, 0, 24, 60), note(0, 24, 24, 62), note(0, 48, 24, 64), note(0, 72, 24, 65)])
    m0 = notes_of(measures(xml)[0])
    assert sorted({n['voice'] for n in m0}) == ['1', '2']
    assert {n['stem'] for n in m0 if n['voice'] == '1' and not n['rest']} == {'up'}
    assert {n['stem'] for n in m0 if n['voice'] == '2' and not n['rest']} == {'down'}
    assert m0[0]['name'] == 'C' and m0[0]['type'] == 'whole' and m0[0]['voice'] == '1'


def test_at_most_four_voices_then_the_nearest_sounding_note_is_cut():
    rows = [note(0, 0, 96, 84), note(0, 0, 72, 79), note(0, 0, 48, 76), note(0, 0, 36, 72), note(0, 24, 24, 70)]
    xml = render(rows)
    m0 = notes_of(measures(xml)[0])
    assert sorted({n['voice'] for n in m0}) == ['1', '2', '3', '4']
    cut = [n for n in m0 if n['name'] == 'C' and n['octave'] == '5']
    assert cut[0]['type'] == 'quarter'    # the C5 was shortened to make room for Bb4
    stems = {n['voice']: n['stem'] for n in m0 if not n['rest']}
    assert stems == {'1': 'up', '2': 'down', '3': 'up', '4': 'down'}


def test_same_onset_different_durations_are_two_voices():
    xml = render([note(0, 0, 96, 48), note(0, 0, 24, 60), note(0, 24, 24, 62), note(0, 48, 48, 64)])
    m0 = notes_of(measures(xml)[0])
    assert sorted({n['voice'] for n in m0}) == ['1', '2']


def test_chords_merge_and_off_grid_values_are_snapped():
    xml = render([note(0, 0, 24, 60), note(0, 0, 24, 64), note(0, 0, 24, 67), note(0, 25, 5, 69), note(0, 31, 7, 71)])
    assert '<time-modification>' not in xml
    m0 = measures(xml)[0]
    assert len(m0.findall('note/chord')) == 2
    assert all(n['type'] in ('quarter', 'eighth', '16th', '32nd', 'half', 'whole') for n in notes_of(m0))


def test_triplet_beat_renders_as_one_tuplet():
    xml = render([note(0, 0, 8, 60), note(0, 8, 8, 62), note(0, 16, 8, 64), note(0, 24, 24, 65)])
    m0 = measures(xml)[0]
    assert len(m0.findall('note/time-modification')) == 3
    starts = [t for t in m0.findall('note/notations/tuplet') if t.get('type') == 'start']
    assert len(starts) == 1
    assert 'print-object="no"' not in xml


def test_beams_and_no_stems_in_single_voice_bars():
    xml = render([note(0, p, 12, 60 + i) for i, p in enumerate(range(0, 96, 12))])
    m0 = measures(xml)[0]
    assert len(m0.findall('note/beam')) == 8
    assert not m0.findall('note/stem')


def test_measures_stay_aligned_across_staves():
    rows = [('INSTR_piano', '', '', '', 'VOICE_1'), note(0, 72, 48, 60, 0), note(1, 0, 24, 40, 1), note(2, 0, 24, 41, 1)]
    xml = tokens_to_score([HEADER[0], HEADER[1], rows[0], HEADER[2], ('M_0', 'POS_0', 'NUM_4', 'DEN_4', ''), *rows[1:], ('EOS', '', '', '', '')])
    divisions = int(re.search(r'<divisions>(\d+)', xml).group(1))
    root = ET.fromstring(xml)
    parts = root.findall('part')
    assert len(parts) == 2 and all(len(p.findall('measure')) == 3 for p in parts)
    for p in parts:
        for m in p.findall('measure'):
            assert all(s == 4 for s in bar_sum(m, divisions).values())


def test_round_trip_reads_back_the_same_events():
    import music21
    from tokenizer import tokenize_score_as_text
    rows = [note(0, 60, 48, 66), note(0, 0, 24, 60), note(1, 12, 24, 62), note(1, 36, 36, 64)]
    xml = render(rows)
    score = music21.converter.parseData(xml, format='musicxml')
    score.stripTies(inPlace=True)
    back = [row for row in tokenize_score_as_text(score) if row[3].startswith('PITCH')]
    from tokenizer import resolve_bars
    got = sorted((r[1], r[2], r[3]) for r in resolve_bars(back) if r[0] in ('M_0', 'M_1'))
    want = sorted((r[1], r[2], r[3]) for r in rows)
    assert got == want
