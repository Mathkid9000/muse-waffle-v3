from __future__ import annotations
import music21, os, math, json, re, itertools, copy
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
from tqdm import tqdm

# file paths
load_dotenv('env.env')
STORAGE_DIR = os.getenv('STORAGE_DIR')
VOCAB_LIST_PATH = Path(STORAGE_DIR) / 'vocab.json'
VOCAB_LIST_PLOT_PATH = Path(STORAGE_DIR) / 'vocab_plot.json'
DATASET_PATH = Path(STORAGE_DIR) / 'data.npz'

# params
SUBDIVISIONS_PER_QUARTER = 24
CHORD_QUALITIES = ('', 'm', 'm7', '7', 'o7', 'dim', '7sus', 'sus2', 'sus', '7+', '+', 'maj7')
CHORD_SLOTS = { 0: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 10: 3, 11: 3 } # semitones above root -> root/third/fifth/seventh

# TODO:
# 
# - fix barlines not rendering correctly
# - fix weird pick up measure lengths
# - add grace notes
#  

def preprocess_dataset():
    openscore_path = Path(STORAGE_DIR) / "data/openscore"
    files = [ f for f in openscore_path.iterdir() if f.is_file() and f.suffix == ".mxl" ]

    # build vocab
    token_seqs = []
    idx = -1
    for path in tqdm(files, "Processing..."):
        idx += 1
        found_error = False
        score = music21.converter.parse(path)
        # process 0 first
        try:
            tokenized, chord_toks = tokenize_score_as_text(score, return_chords=True)
            token_seqs.append(tokenized)
        except Exception as e:
            print(f'Error: {e}. Skipping: {idx}')
            found_error = True

        if not found_error:
            # process the rest
            for interval in tqdm(range(-5, 7), "Intervals", total=12, leave=False):
                if interval == 0: continue
                transposed = score.transpose(music21.interval.Interval(interval))
                tokenized = tokenize_score_as_text(transposed, transpose_chord_toks=(chord_toks, interval))
                token_seqs.append(tokenized)

    vocab = build_vocab_list(token_seqs, plot=True)
    save_vocab_list(vocab)
    
    # convert to dataset
    tokenized = [ tokenize_text_to_ids(seq, vocab) for seq in token_seqs ]
    np.savez(DATASET_PATH, *tokenized)


def load_dataset():
    loaded = np.load(DATASET_PATH)
    return [ loaded[f] for f in loaded ]


def tokenize_text_to_ids(quint_tokens, vocab_list):
    quint_ids = []
    for tok_seq in quint_tokens:
        quint_ids.append((vocab_list[0].index(tok_seq[0]), vocab_list[1].index(tok_seq[1]), vocab_list[2].index(tok_seq[2]), vocab_list[3].index(tok_seq[3]), vocab_list[4].index(tok_seq[4])))
    return quint_ids


def detokenize_ids(quint_ids, vocab_list):
    quint_toks = []
    for tok_seq in quint_ids:
        quint_toks.append((vocab_list[0][tok_seq[0]], vocab_list[1][tok_seq[1]], vocab_list[2][tok_seq[2]], vocab_list[3][tok_seq[3]], vocab_list[4][tok_seq[4]]))
    return quint_toks


def analyze_chord(weights, voicing, key_ctx):
    # name a window's pitch classes (weights = how long each sounds, voicing = the lowest pitch it
    # sounds at, kept so the real bass still drives root detection) as (degree, root, quality), or
    # None if they make no chord we have a token for. Only the heaviest tone of each slot above the
    # root survives, so 9ths/11ths/13ths are stripped and a 7th is the richest possible result, and
    # whatever the slots can't settle (eg. E and F over C) falls to whichever sounds longer.
    root = music21.chord.Chord(sorted(voicing.values())).root().pitchClass
    kept = {}
    # longest sounding tone claims each slot, and the lower voice wins an outright tie
    for pc in sorted(weights, key=lambda pc: (-weights[pc], voicing[pc])):
        kept.setdefault(CHORD_SLOTS.get((pc - root) % 12, -1), pc)
    kept.pop(-1, None)
    # two lone tones spell a chord only when they form a real third; a second or a fourth between
    # them is a melodic collision rather than a suspension, and a lone root+seventh is nothing
    if len(kept) == 2 and (kept.get(1, root) - root) % 12 not in (3, 4): return None
    # music21 can't name a chord that has no fifth, but an omitted fifth doesn't change what the
    # chord is, so stand one in, taking its alteration from the key so a leading tone gets the
    # diminished fifth it wants (B in C major -> F, not F#); a real fifth would fill the slot itself
    if 2 not in kept:
        scale = { p.pitchClass for p in key_ctx.pitches }
        fifth = next((f for f in (7, 6, 8) if (root + f) % 12 in scale), 7)
        kept[2] = (root + fifth) % 12
        voicing[kept[2]] = voicing[root] + fifth
    # the 7th chord first, then the bare triad if that wasn't nameable
    for subset in dict.fromkeys([ tuple(sorted(kept)), tuple(slot for slot in sorted(kept) if slot != 3) ]):
        if len(subset) < 3: continue
        c = music21.chord.Chord(sorted(voicing[kept[slot]] for slot in subset))
        # flip enharmonics (eg. F,G#,C -> F,Ab,C) to find the spelling with the shortest chord symbol
        c = min(
            (music21.chord.Chord([ p.getEnharmonic() if flip else p for p, flip in zip(c.pitches, flips) ]) for flips in itertools.product(*[ (False, True) if p.accidental else (False,) for p in c.pitches ])),
            key=lambda cand: len(music21.harmony.chordSymbolFigureFromChord(cand, includeChordType=True)[0])
        )
        figure, kind = music21.harmony.chordSymbolFigureFromChord(c, includeChordType=True)
        if not kind: continue
        quality = re.split(r'/|add|omit|dim5', figure[len(c.root().name):])[0]
        if quality not in CHORD_QUALITIES: continue
        return music21.roman.romanNumeralFromChord(c, key_ctx).scaleDegree, c.root().midi % 12, quality
    return None


def chord_walker(chordified_measure, key_ctx):
    # walk the measure in ever wider windows (half quarter -> quarter -> half -> whole bar),
    # analyzing a window only when nothing finer inside it already resolved, so wide spans just
    # pool the leftovers of a slow harmonic rhythm instead of blurring chords that were found
    bar_ql = float(chordified_measure.barDuration.quarterLength)
    # the 4th field is the tuplet's plain note value (eg. 0.5 for triplet 8ths), 0 when not a tuplet
    spans = [ (float(c.offset), float(c.offset + c.quarterLength), { p.midi for p in c.pitches }, float(c.duration.tuplets[0].durationNormal.quarterLength) if c.duration.tuplets and c.duration.tuplets[0].durationNormal else 0) for c in chordified_measure.getElementsByClass(music21.chord.Chord) ]
    chords = {}
    steps = [ s for s in dict.fromkeys((0.5, 1, 2, bar_ql)) if s <= bar_ql ]
    # every breadth is walked twice: once demanding a full window, then again settling for a bare
    # root+third dyad, so thin two-note evidence only fills gaps nothing better could explain
    for step, min_pcs in [ (s, 3) for s in steps ] + [ (s, 2) for s in steps ]:
        for pos in ( i * step for i in range(math.ceil(bar_ql / step)) ):
            if any(pos < o + s and o < pos + step for o, (_, s) in chords.items()): continue
            # a tuplet this wide or wider spills past these window edges, so it would get chopped
            # mid-group; leave the whole window to a wider walk that can take the group in one piece
            if any(tuplet >= step and min(end, pos + step) > max(start, pos) for start, end, _, tuplet in spans): continue
            weights, voicing = {}, {}
            for start, end, midis, _ in spans:
                overlap = min(end, pos + step) - max(start, pos)
                if overlap <= 0: continue
                for midi in midis:
                    pc = midi % 12
                    weights[pc] = weights.get(pc, 0) + overlap
                    voicing[pc] = min(voicing.get(pc, midi), midi)
            if len(weights) < min_pcs: continue
            named = analyze_chord(weights, voicing, key_ctx)
            if named: chords[pos] = (named, step)
    return [ (pos, *chords[pos][0]) for pos in sorted(chords) ]


def tokenize_score_as_text(score: music21.stream.Score, return_chords=False, transpose_chord_toks=None):
    def tokify(classifier, tok, lower=True):
        tok_str = str(tok).replace('.0', '')
        return f'{classifier}_{tok_str.lower() if lower else tok_str}'
    def tok_priority(x):
        priority = int(x[0].split('_')[-1]) * 100000 + int(x[1].split('_')[-1]) + ((int(x[4].split('_')[-1]) + 1) / len(instr_toks) if x[4].startswith('VOICE') else 0)
        if x[2].startswith('KEY'): priority -= 5 # key
        if x[2].startswith('NUM'): priority -= 4 # time
        if x[2].startswith('TEMPO'): priority -= 3 # time
        if x[2].startswith('CLEF'): priority -= 2 # clef
        if x[2].startswith('BAR'): priority -= 1 # bar line
        if x[2].startswith('DEG'): priority -= 0.5 # chord
        if x[2].startswith('DYN'): priority -= 0.1 # time
        return priority
    # construct instruments
    instr_toks = []
    for idx, part in enumerate(score.parts):
        name = part.getInstrument().instrumentName
        instr_toks.append((tokify('INSTR', name), '', '', '', tokify('VOICE', idx)))

    chordified = score.chordify()
    score_toks = []
    CHORD_TOKS_RECORD = []
    seen_bars = set()
    last_chord = (None, None)
    for voz_idx, part in tqdm(enumerate(score.parts), "Scanning voices...", leave=False, total=len(score.parts)):
        measure_offset = 0  
        for measure in tqdm(part.getElementsByClass(music21.stream.Measure), "Scanning measures...", leave=False):
            suffix = measure.measureNumberWithSuffix()
            if 'X' in suffix: measure_offset += 1

            m_num = measure.number + measure_offset if measure.number is not None else 0

            # flatten notes and rests if there are multiple voices
            notesAndRests: list[music21.note.GeneralNote] = []
            if measure.voices:
                for voice in measure.voices:
                    notesAndRests.extend(voice)
            else:
                notesAndRests.extend(measure.notesAndRests)

            # read all notes and convert to text toks
            note_toks = []
            for note in notesAndRests:
                if note.isRest:
                    continue

                pos = note.offset * SUBDIVISIONS_PER_QUARTER
                dur = note.duration.quarterLength * SUBDIVISIONS_PER_QUARTER
                if dur == 0: continue

                for pitch in note.pitches:
                    note_toks.append((tokify('M', m_num), tokify('POS', pos), tokify('DUR', dur), tokify('PITCH', pitch.midi), tokify('VOICE', voz_idx)))

            # read all key sigs
            key_toks = []
            if voz_idx == 0:
                for key_sig in measure.getElementsByClass(music21.key.KeySignature):
                    tonic = key_sig.asKey().tonic.midi % 12
                    pos = key_sig.offset * SUBDIVISIONS_PER_QUARTER
                    key_toks.append((tokify('M', m_num), tokify('POS', pos), tokify('KEY', tonic), '', ''))

            # read all time sigs
            time_toks = []
            if voz_idx == 0:
                for ts in measure.getElementsByClass(music21.meter.TimeSignature):
                    pos = ts.offset * SUBDIVISIONS_PER_QUARTER
                    time_toks.append((tokify('M', m_num), tokify('POS', pos), tokify('NUM', ts.numerator), tokify('DEN', ts.denominator), ''))

            # read all tempo markings (~16 buckets), including text tempos like "Andante"
            tempo_toks = []
            if voz_idx == 0:
                text_tempos = []
                for te in measure.getElementsByClass(music21.expressions.TextExpression):
                    mm = music21.tempo.MetronomeMark(text=te.content)
                    mm.offset = te.offset
                    text_tempos.append(mm)
                for mm in [ *measure.getElementsByClass(music21.tempo.MetronomeMark), *text_tempos ]:
                    if mm.number is None: continue
                    pos = mm.offset * SUBDIVISIONS_PER_QUARTER
                    tempo_toks.append((tokify('M', m_num), tokify('POS', pos), tokify('TEMPO', round(mm.number / 16)), '', ''))

            # read all dynamics
            dynamic_toks = []
            for dyn in measure.getElementsByClass(music21.dynamics.Dynamic):
                pos = dyn.offset * SUBDIVISIONS_PER_QUARTER
                dynamic_toks.append((tokify('M', m_num), tokify('POS', pos), tokify('DYN', dyn.value), '', tokify('VOICE', voz_idx)))

            # read all clefs
            clef_toks = []
            for clef in measure.getElementsByClass(music21.clef.Clef):
                pos = clef.offset * SUBDIVISIONS_PER_QUARTER
                clef_toks.append((tokify('M', m_num), tokify('POS', pos), tokify('CLEF', clef.name), '', tokify('VOICE', voz_idx)))

            # read all bar lines (including repeats); not always on the first staff, so check every part
            bar_toks = []
            for barline in measure.getElementsByClass(music21.bar.Barline):
                pos = barline.offset * SUBDIVISIONS_PER_QUARTER
                if (m_num, pos) in seen_bars: continue
                seen_bars.add((m_num, pos))
                bar_type = f'repeat-{barline.direction}' if isinstance(barline, music21.bar.Repeat) else barline.type
                bar_toks.append((tokify('M', m_num), tokify('POS', pos), tokify('BAR', bar_type), '', ''))

            # walk the measure for chords, analyzed across all staves; skip repeats of the same chord
            chord_toks = []
            if voz_idx == 0 and transpose_chord_toks == None:
                chordified_measure = chordified.measure(m_num)
                if chordified_measure is not None:
                    key_ctx = (measure.getContextByClass(music21.key.KeySignature) or music21.key.KeySignature(0)).asKey()
                    for pos, degree, root, quality in chord_walker(chordified_measure, key_ctx):
                        if (root, quality) == last_chord: continue
                        last_chord = (root, quality)
                        chord_toks.append((tokify('M', m_num), tokify('POS', pos * SUBDIVISIONS_PER_QUARTER), tokify('DEG', degree), tokify('ROOT', root), tokify('QUAL', quality, lower=False)))
                CHORD_TOKS_RECORD.extend(copy.deepcopy(chord_toks))

            # extend all to score_toks
            score_toks.extend(note_toks)
            score_toks.extend(key_toks)
            score_toks.extend(time_toks)
            score_toks.extend(tempo_toks)
            score_toks.extend(dynamic_toks)
            score_toks.extend(clef_toks)
            score_toks.extend(bar_toks)
            score_toks.extend(chord_toks)

    # add chords if non transposed are given
    if transpose_chord_toks != None:
        original_chords, interval = transpose_chord_toks
        for t in original_chords:
            new_root = tokify('ROOT', (int(t[3].split('_')[-1]) + interval) % 12)
            score_toks.append((t[0], t[1], t[2], new_root, t[4]))

    # final sort
    score_toks.sort(key=lambda x: tok_priority(x))
    
    # constructed
    output = [
        ('SOG', '', '', '',''),
        *instr_toks,
        ('SOS', '', '', '',''),
        *score_toks,
        ('EOS', '', '', '', '')
    ]

    with open(Path(STORAGE_DIR) / 'urmom.txt', 'w') as f:
        f.writelines('\n'.join([ str(t) for t in output ]))

    if return_chords:
        return output, CHORD_TOKS_RECORD
    else: return output


def pretty(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print('  ' * indent + str(key))
            pretty(value, indent+1)
        else:
            print('  ' * (indent+1) + str(key) + str(value))


def tokens_to_score(tokens):
    def parse_tok(tok):
        output = []
        for t in tok:
            try:
                output.append(int(t.split('_')[-1]))
            except:
                output.append(t.split('_')[-1])
        return output[0], output[1], output[2], output[3], output[4]
    def find_overlapping(voice, note_m, note_pos, note_dur):
        # returns the next available sub_voice
        overlapping_subvoices = []

        original_start = note_pos
        original_end = note_pos + note_dur
        for subvoice in notes[voice].keys():
            for m, pos, dur in notes[voice][subvoice].keys():
                if m != note_m: continue
                start = pos
                end = pos + dur
                if max(start, original_start) < min(end, original_end):
                    overlapping_subvoices.append(subvoice)
        if len(overlapping_subvoices) == 0: return None
        if set(sorted(overlapping_subvoices)) == set(range(max(overlapping_subvoices) + 1)): return max(overlapping_subvoices) + 1
        for i in range(max(overlapping_subvoices) + 1):
            if i not in overlapping_subvoices:
                return i
    def spell_pitch(midi, sharps):
        pitch = music21.pitch.Pitch(midi=midi)
        if sharps is None or pitch.accidental is None:
            return pitch
        prefers_sharps = sharps >= 0
        is_sharp = pitch.accidental.alter > 0
        if prefers_sharps != is_sharp:
            return pitch.getEnharmonic()
        return pitch

    score = music21.stream.Score()
    parts = []
    general_score_elements: dict[int, list[music21.base.Music21Object]] = {} # measure num -> objects
    voice_score_elements: dict[tuple[int, int], list[music21.base.Music21Object]] = {} # (voice,measure) -> objects
    notes: dict[int, dict[int, dict[tuple[int, int, int], list[int]]]] = {} # notes[voice][sub_voice][(measure,pos,dur)] = pitches
    measure_key_sharps: dict[int, int] = {} # measure num -> sharps in effect at that measure

    for tok in tokens:
        # add instruments
        if tok[0].startswith('INSTR'):
            part = music21.stream.Part()
            part.insert(0, music21.instrument.fromString(tok[0].split('_')[-1]))
            parts.append(part)
        # add notes
        if tok[3].startswith('PITCH'):
            m, pos, dur, pitch, voz = parse_tok(tok)
            if voz not in notes:
                notes[voz] = {}

            # case 1: there is already a note in that exact pos with exact dur
            found = False
            for subvoice in notes[voz].keys():
                if (m, pos, dur) in notes[voz][subvoice]:
                    notes[voz][subvoice][(m,pos,dur)].append(pitch)
                    found = True
                    break
            if found: continue
            # case 2: theres an overlapping note
            overlapping = find_overlapping(voz, m, pos, dur)
            if overlapping != None:
                if overlapping not in notes[voz]:
                    notes[voz][overlapping] = {}
                notes[voz][overlapping][(m, pos, dur)] = [ pitch ]
                continue
            # case 3: there are no overlapping notes
            if 0 not in notes[voz]:
                notes[voz][0] = {}
            notes[voz][0][(m, pos, dur)] = [ pitch ]
        # add key
        if tok[2].startswith('KEY'):
            m, pos, tonic, _, _ = parse_tok(tok)
            pitch = music21.pitch.Pitch(midi=tonic)

            possible = music21.analysis.enharmonics.EnharmonicSimplifier([pitch]).allSpellings[0]
            smallest = (possible[0], music21.key.Key(possible[0]).sharps)
            for spelling in possible[1:]:
                key = music21.key.Key(spelling)
                if abs(key.sharps) < abs(smallest[1]):
                    smallest = (spelling, key.sharps)

            pitch = music21.analysis.enharmonics.EnharmonicSimplifier([pitch], ruleClass=music21.analysis.enharmonics.ChordEnharmonicScoreRules).bestPitches()[0]
            key_sig = music21.key.KeySignature(smallest[1])
            key_sig.offset = pos / SUBDIVISIONS_PER_QUARTER
            if m not in general_score_elements:
                general_score_elements[m] = []
            general_score_elements[m].append(key_sig)
            measure_key_sharps[m] = smallest[1]
        # add time sig
        if tok[2].startswith('NUM'):
            m, pos, num, den, _ = parse_tok(tok)
            ts = music21.meter.TimeSignature(f'{num}/{den}')
            ts.offset = pos / SUBDIVISIONS_PER_QUARTER
            if m not in general_score_elements:
                general_score_elements[m] = []
            general_score_elements[m].append(ts)
        # add tempo
        if tok[2].startswith('TEMPO'):
            m, pos, bucket, _, _ = parse_tok(tok)
            mm = music21.tempo.MetronomeMark(number=bucket * 16)
            mm.offset = pos / SUBDIVISIONS_PER_QUARTER
            if m not in general_score_elements:
                general_score_elements[m] = []
            general_score_elements[m].append(mm)
        # add dynamic
        if tok[2].startswith('DYN'):
            m, pos, dyn, _, voz = parse_tok(tok)
            d = music21.dynamics.Dynamic(dyn)
            d.offset = pos / SUBDIVISIONS_PER_QUARTER
            if (voz, m) not in voice_score_elements:
                voice_score_elements[(voz, m)] = []
            voice_score_elements[(voz, m)].append(d)
        # add clef
        if tok[2].startswith('CLEF'):
            m, pos, clef_name, _, voz = parse_tok(tok)
            clef = music21.clef.clefFromString(clef_name)
            clef.offset = pos / SUBDIVISIONS_PER_QUARTER
            if (voz, m) not in voice_score_elements:
                voice_score_elements[(voz, m)] = []
            voice_score_elements[(voz, m)].append(clef)
        # add bar line
        if tok[2].startswith('BAR'):
            m, pos, bar_type, _, _ = parse_tok(tok)
            barline = music21.bar.Repeat(direction=bar_type.split('-')[-1]) if bar_type.startswith('repeat-') else music21.bar.Barline(bar_type)
            barline.offset = pos / SUBDIVISIONS_PER_QUARTER
            if m not in general_score_elements:
                general_score_elements[m] = []
            general_score_elements[m].append(barline)
        # add chord
        if tok[2].startswith('DEG'):
            m, pos, _, root, quality = parse_tok(tok)
            p = music21.pitch.Pitch()
            p.pitchClass = root
            cs = music21.harmony.ChordSymbol(p.name + str(quality))
            # force kind='other': some of our kind names (eg. 'suspended-fourth-seventh') aren't valid
            # MusicXML kind-value enums, so readers ignore chordKindStr and show the raw kind name instead;
            # 'other'+chordKindStr is always spec-valid and reliably displays our concise abbreviation
            if quality == '7sus': cs.chordKind = 'other'
            cs.chordKindStr = str(quality)
            cs.offset = pos / SUBDIVISIONS_PER_QUARTER
            if (0, m) not in voice_score_elements:
                voice_score_elements[(0, m)] = []
            voice_score_elements[(0, m)].append(cs)

    # carry each key signature forward until the next one changes it
    if measure_key_sharps:
        note_measure_nums = [
            m for voz in notes for subvoice in notes[voz] for (m, _, _) in notes[voz][subvoice].keys()
        ]
        last_measure = max(note_measure_nums + list(measure_key_sharps.keys()))
        current_sharps = 0
        filled_key_sharps = {}
        for m in range(last_measure + 1):
            if m in measure_key_sharps:
                current_sharps = measure_key_sharps[m]
            filled_key_sharps[m] = current_sharps
        measure_key_sharps = filled_key_sharps

    # concatenate all notes into measures
    measures: dict[tuple[int, int], music21.stream.Measure] = {} # (voice, measure) = measure
    for voz_idx, part in enumerate(parts):
        if len(notes[voz_idx].keys()) == 1:
            for m, pos, dur in notes[voz_idx][0]:
                pitches = notes[voz_idx][0][(m, pos, dur)]
                sharps = measure_key_sharps.get(m)
                if len(pitches) == 1:
                    generalNote = music21.note.Note(spell_pitch(pitches[0], sharps))
                else:
                    generalNote = music21.chord.Chord([ spell_pitch(i, sharps) for i in pitches ])
                generalNote.duration = music21.duration.Duration(dur / SUBDIVISIONS_PER_QUARTER)
                generalNote.offset = pos / SUBDIVISIONS_PER_QUARTER

                if (voz_idx, m) not in measures:
                    measures[(voz_idx, m)] = music21.stream.Measure(number=m)
                measures[(voz_idx, m)].insert(generalNote.offset, generalNote)
        else:
            subvoices: dict[int, dict[int, music21.stream.Voice]] = {}
            for subvoice_idx in notes[voz_idx].keys():
                for m, pos, dur in notes[voz_idx][subvoice_idx]:
                    pitches = notes[voz_idx][subvoice_idx][(m, pos, dur)]
                    sharps = measure_key_sharps.get(m)
                    if len(pitches) == 1:
                        generalNote = music21.note.Note(spell_pitch(pitches[0], sharps))
                    else:
                        generalNote = music21.chord.Chord([ spell_pitch(i, sharps) for i in pitches ])
                    generalNote.duration = music21.duration.Duration(dur / SUBDIVISIONS_PER_QUARTER)
                    generalNote.offset = pos / SUBDIVISIONS_PER_QUARTER
                    if m not in subvoices:
                        subvoices[m] = { }
                    if subvoice_idx not in subvoices[m]:
                        subvoice = music21.stream.Voice()
                        subvoices[m][subvoice_idx] = subvoice
                    subvoices[m][subvoice_idx].insert(generalNote.offset, generalNote)
            for m in subvoices.keys():
                if (voz_idx, m) not in measures:
                    measures[(voz_idx, m)] = music21.stream.Measure(number=m)
                for subvoice_idx in subvoices[m].keys():
                    measures[(voz_idx, m)].insert(0, subvoices[m][subvoice_idx])

    # concatenate measures into parts
    min_measure = min([ m for _, m in measures.keys() ])
    for voz_idx in range(len(parts)):
        for m in range(min_measure, max([ k[1] for k in list(measures.keys()) ]) + 1):
            if (voz_idx, m) in measures:
                measure = measures[(voz_idx, m)]
            else:
                measure = music21.stream.Measure(number=m)

            if m in general_score_elements:
                for element in general_score_elements[m]:
                    if isinstance(element, music21.bar.Barline):
                        if element.offset == 0:
                            measure.leftBarline = element
                        else:
                            measure.rightBarline = element
                    else:
                        measure.insert(element)
            if (voz_idx, m) in voice_score_elements:
                for element in voice_score_elements[(voz_idx, m)]:
                    measure.insert(element)

            # formatting
            sharps = measure_key_sharps.get(m)
            key_context = music21.key.KeySignature(sharps) if sharps is not None else True
            measure.makeAccidentals(inPlace=True, overrideStatus=True, useKeySignature=key_context)
            if measure.voices:
                for voice in measure.voices:
                    voice.makeRests(inPlace=True)
            elif measure.notesAndRests:
                measure.makeRests(inPlace=True)
            else:
                measure.makeRests(inPlace=True, timeRangeFromBarDuration=True)
            parts[voz_idx].append(measure)

    # concatenate parts into score
    for part in parts:
        # formatting
        part.makeBeams(inPlace=True)
        score.insert(part)

    # makeNotation=False: the default export path re-fills every measure/voice up
    # to a guessed bar duration (fillGaps+timeRangeFromBarDuration), which pads
    # incomplete measures back up instead of keeping their real, shorter length.
    score.write("musicxml", Path(STORAGE_DIR) / "output.mxl", makeNotation=True)


def build_vocab_list(token_sequences, plot=False):
    vocab = [[], [], [], [], []]
    histogram = [{}, {}, {}, {}, {}]
    for seq in token_sequences:
        for quint_tok in seq:
            for idx, tok in enumerate(quint_tok):
                if tok not in vocab[idx]:
                    vocab[idx].append(tok)
                    histogram[idx][tok] = 0
                histogram[idx][tok] += 1

    # plot freqs
    if plot:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 3, figsize=(12, 8), layout='constrained')
        for idx, histo in enumerate(histogram):
            pairs = sorted(zip(histo.keys(), histo.values()), key=lambda x: x[0], reverse=True)
            keys, values = zip(*pairs)
            axs[idx % 2, idx // 2].bar(keys, values, color=['red', 'orange', 'lime', 'cyan', 'violet'][idx], edgecolor='black')
            axs[idx % 2, idx // 2].tick_params(axis='x', rotation=90)
            axs[idx % 2, idx // 2].set_title(f"HEAD {idx + 1}")
        axs[1,2].hist([ len(seq) for seq in token_sequences ], bins=200, edgecolor='black', color='yellow')
        plt.tight_layout()
        plt.show()
        with open(VOCAB_LIST_PLOT_PATH, 'w') as f:
            json.dump(histogram, f)
    return vocab


def save_vocab_list(vocab):
    with open(VOCAB_LIST_PATH, 'w') as f:
        json.dump(vocab, f)


def load_vocab_list():
    with open(VOCAB_LIST_PATH, 'r') as f:
        return json.load(f)


if __name__  == "__main__":
    # preprocess_dataset()

    # validate
    rng = np.random.default_rng()
    quint_id_seqs = load_dataset()
    vocab_list = load_vocab_list()
    detokenized = [ detokenize_ids(seq, vocab_list) for seq in quint_id_seqs ]
    chosen = detokenized[rng.choice(len(detokenized))]
    tokens_to_score(chosen)
