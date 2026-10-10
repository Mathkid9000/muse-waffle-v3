"""Clefs, key and time signatures and tempo marks are written where they change and nowhere else,
however the bars around them were generated, pasted or removed -- and a template chord may name
its root absolutely or by numeral."""

import xml.etree.ElementTree as ET

import music21 as m21
import pytest

import notation
from tokenizer import tokens_to_score
from api.routers.edit.rm_measure import _run_edit as rm_measures
from api.routers.edit.write_score import _run_edit as write_score
from api.routers.templates.fill_template import _canvas, chord_root
from api.routers.templates.score_to_template import _run_conversion as score_to_template
from api.routers.templates.template_to_xml import _run_conversion as template_to_xml
from api.routers.templates.templateSchemas import ChordTemplateItem, FillTemplateRequest
from api.schemas import EditRmMeasureRequest, EditWriteScoreRequest, XmlResponse

HEADER = [('SOG', '', '', '', ''), ('INSTR_piano', '', '', '', 'VOICE_0'), ('SOS', '', '', '', '')]
MARKS = ('clef', 'key', 'time', 'metronome')


def score(sharps=0, time='4/4', bpm=96, bars=2, pitch='D4', clef=None):
    s, p = m21.stream.Score(), m21.stream.Part()
    p.insert(0, m21.instrument.Piano())
    for n in range(1, bars + 1):
        m = m21.stream.Measure(number=n)
        if n == 1:
            for el in (clef or m21.clef.TrebleClef(), m21.key.KeySignature(sharps),
                       m21.meter.TimeSignature(time), m21.tempo.MetronomeMark(number=bpm)):
                m.insert(0, el)
        m.insert(0, m21.note.Note(pitch, quarterLength=m21.meter.TimeSignature(time).barDuration.quarterLength))
        p.append(m)
    s.insert(0, p)
    return notation.finish(s)


def marks(xml):
    """Per measure of the first part, the state-setting symbols written in it."""
    out = []
    for measure in ET.fromstring(xml).find('part').findall('measure'):
        found = {}
        if measure.find('attributes/clef') is not None: found['clef'] = measure.findtext('attributes/clef/sign')
        if measure.find('attributes/key') is not None: found['key'] = int(measure.findtext('attributes/key/fifths'))
        if measure.find('attributes/time') is not None:
            found['time'] = measure.findtext('attributes/time/beats') + '/' + measure.findtext('attributes/time/beat-type')
        if measure.find('.//metronome') is not None: found['metronome'] = int(measure.findtext('.//metronome/per-minute'))
        out.append(found)
    return out


# ---------------------------------------------------------------- pasting and removing bars

def test_paste_at_end_restates_nothing():
    out = marks(write_score(EditWriteScoreRequest(xml=score(2), source_xml=score(2, pitch='A4'), measure_start_num=3)))
    assert set(out[0]) == set(MARKS)
    assert out[1:] == [{}, {}, {}]


def test_paste_at_end_keeps_what_differs():
    out = marks(write_score(EditWriteScoreRequest(xml=score(2), source_xml=score(-1, bpm=120), measure_start_num=3)))
    assert out[2] == {'key': -1, 'metronome': 120}
    assert out[3] == {}


def test_paste_mid_score_leaves_the_tail_in_its_own_key_and_meter():
    out = marks(write_score(EditWriteScoreRequest(
        xml=score(2, bars=4), source_xml=score(-3, time='3/4', bars=1, clef=m21.clef.BassClef()), measure_start_num=2)))
    assert out[1] == {'clef': 'F', 'key': -3, 'time': '3/4'}
    assert out[2] == {'clef': 'G', 'key': 2, 'time': '4/4'}
    assert out[3] == {}


def test_deleting_the_first_bar_keeps_its_signatures():
    out = marks(rm_measures(EditRmMeasureRequest(xml=score(2, bars=3), measure_start_num=1, measure_end_num=1)))
    assert out == [{'clef': 'G', 'key': 2, 'time': '4/4', 'metronome': 96}, {}]


def test_deleting_a_middle_bar_adds_nothing():
    out = marks(rm_measures(EditRmMeasureRequest(xml=score(2, bars=3), measure_start_num=2, measure_end_num=2)))
    assert set(out[0]) == set(MARKS) and out[1] == {}


# ---------------------------------------------------------------- generated rows

def test_rows_restating_the_opening_are_not_written_again():
    opening = [('M_0', 'POS_0', 'KEY_2', '', ''), ('M_0', 'POS_0', 'NUM_4', 'DEN_4', ''),
               ('M_0', 'POS_0', 'TEMPO_6', '', ''), ('M_0', 'POS_0', 'CLEF_treble', '', 'VOICE_0')]
    note = lambda m: (f'M_{m}', 'POS_0', 'DUR_96', 'PITCH_62', 'VOICE_0')
    again = [(f'M_2', *row[1:]) for row in opening]
    out = marks(tokens_to_score([*HEADER, *opening, note(0), note(1), *again, note(2),
                                 ('M_3', 'POS_0', 'KEY_7', '', ''), note(3), ('EOS', '', '', '', '')]))
    assert set(out[0]) == set(MARKS)
    assert out[1:] == [{}, {}, {'key': 1}]


def test_generation_header_skips_what_the_prefix_already_set():
    from ml.api import _header_rows, _in_effect

    prefix = [*HEADER, ('M_SAME', 'POS_0', 'KEY_2', '', ''), ('M_SAME', 'POS_0', 'NUM_4', 'DEN_4', ''),
              ('M_SAME', 'POS_0', 'TEMPO_6', '', ''), ('M_NEXT', 'POS_0', 'KEY_7', '', '')]
    # key index 8 is two sharps (KEY_2), which the prefix has since left for KEY_7
    assert _header_rows(8, 2, 6, 'M_NEXT', _in_effect(prefix)) == [('M_NEXT', 'POS_0', 'KEY_2', '', '')]
    assert _header_rows(7, 2, 6, 'M_NEXT', _in_effect(prefix)) == []
    assert len(_header_rows(7, 2, 6, 'M_NEXT')) == 3


# ---------------------------------------------------------------- template chords

@pytest.mark.parametrize('root,name', [
    ('G', 'G'), ('Bb', 'B-'), ('F#', 'F#'), ('b', 'B'), ('ii', 'G'), ('V', 'C'), ('bVII', 'E-'), ('2', 'G'),
])
def test_chord_root_reads_note_names_and_numerals(root, name):
    assert chord_root(root, m21.key.Key('F')).name == name


@pytest.mark.parametrize('root', [None, '', 'Gm7', 'H', 'nonsense'])
def test_chord_root_unreadable_is_blank(root):
    assert chord_root(root, m21.key.Key('F')) is None


def test_legacy_chord_fields_load_as_root():
    assert ChordTemplateItem(degree='bVII', rootNumeral='7', quality='dom7').root == 'bVII'
    assert ChordTemplateItem(rootNumeral='2').root == 'II'
    assert ChordTemplateItem(root='G', degree='ii').root == 'G'


def template(*chords):
    return FillTemplateRequest(template=[
        dict(kind='instrument', instrument='piano', voice_idx=0, measure=1, pos=0),
        dict(kind='key', key_tonic='F', key_mode='major', measure=1, pos=0),
        dict(kind='note', measure=1, pos=0, duration=4, pitch=70, voice_idx=0),
        *(dict(kind='chord', measure=1, pos=0, **chord) for chord in chords),
    ]).template


def test_absolute_and_numeral_chords_reach_the_model_as_the_same_row():
    rows = [row for row in _canvas(template(dict(root='G', quality='min7'), dict(root='ii', quality='min7')))
            if row[2].startswith('DEG')]
    assert rows == [('M_SAME', 'POS_0', 'DEG_2', 'ROOT_7', 'QUAL_m7')] * 2


def test_unreadable_chord_root_is_masked_not_refused():
    row = next(row for row in _canvas(template(dict(root='Gm7', quality='min7'))) if row[4].startswith('QUAL'))
    assert row[2:4] == ('MASK', 'MASK')


@pytest.mark.parametrize('root,label', [('Bb', 'Bbm7'), ('ii', 'iim7')])
def test_template_preview_labels_a_chord_as_written(root, label):
    xml = template_to_xml(template(dict(root=root, quality='min7')))
    assert label in [w.text for w in ET.fromstring(xml).iter('words')]


def test_every_quality_previews_under_an_absolute_root():
    for quality in ChordTemplateItem.model_fields['quality'].annotation.__args__[0].__args__:
        template_to_xml(template(dict(root='Eb', quality=quality)))


def test_score_to_template_writes_numerals():
    s, p = m21.stream.Score(), m21.stream.Part()
    m = m21.stream.Measure(number=1)
    for el in (m21.key.KeySignature(-1), m21.meter.TimeSignature('4/4'), m21.harmony.ChordSymbol('Gm7'),
               m21.note.Note('B-4', quarterLength=4)):
        m.insert(0, el)
    p.append(m)
    s.insert(0, p)
    chord = next(it for it in score_to_template(XmlResponse(xml=notation.finish(s))) if it.kind == 'chord')
    assert (chord.root, chord.quality) == ('ii', 'min7')
