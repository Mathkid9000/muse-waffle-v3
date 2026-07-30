from __future__ import annotations
import music21, os, math, json, re, itertools, gzip, zlib, zipfile
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm
from helpers import STORAGE_DIR, DATA_DIR, DATASET_PATH, VOCAB_LIST_PATH, VOCAB_LIST_PLOT_PATH, TOKEN_CACHE_DIR

# params
SUBDIVISIONS_PER_QUARTER = 24
CHORD_QUALITIES = ('', 'm', 'm7', '7', 'o7', 'ø7', 'dim', '7sus', 'sus2', 'sus', '7+', '+', 'maj7')
CHORD_SLOTS = { 0: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 10: 3, 11: 3 } # semitones above root -> root/third/fifth/seventh
TRANSPOSE_INTERVALS = range(-5, 7)
MAX_BAR_SKIP = 16      # any wider gap between consecutive rows is musically just "big gap"
MIN_PITCH, MAX_PITCH = 21, 108   # piano range; keeps transposition inside 16..114
MIN_TEMPO, MAX_TEMPO = 10, 400   # anything outside is typesetter garbage (corpus has 1.2M "BPM")
MAX_VOICES = 17        # VOICE_0..VOICE_16; parts past that merge into the last voice

# TODO:
# 
# - fix barlines not rendering correctly
# - fix weird pick up measure lengths
# - add grace notes
#  

def tokenize_file(path):
    # one score, start to finish, in whatever process the pool hands it to. Errors come back as a
    # value rather than killing the worker, since a few hundred of the corpus's files fail to parse.
    try:
        return tokenize_score_as_text(music21.converter.parse(path)), None
    except Exception as e:
        return None, f'{type(e).__name__}: {e}'


def score_files():
    return [ f
             for dataset in ('openscore', 'pdmx')
             for f in (DATA_DIR / dataset).iterdir()
             if f.is_file() and f.suffix == '.mxl' ]


def score_name(file):
    return f'{file.parent.name}/{file.name}'


def preprocess_dataset(workers=None):
    # two stages, because parsing the full PDMX takes longer than a remote session lasts: stage one
    # parses scores to token text in a resumable on-disk cache, stage two turns that cache into
    # vocab.json + data.npz. Rerunning is always safe -- stage one picks up where it left off and
    # stage two is a few minutes over the finished cache. Delete TOKEN_CACHE_DIR to force a reparse.
    files = score_files()
    tokenize_to_cache(files, workers)
    build_dataset_from_cache()


# ---------------------------------------------------------------- stage 1: parse -> token cache

CACHE_SHARD_SCORES = 20000   # scores per shard; a shard is closed and never touched again
CACHE_FLUSH_SCORES = 500     # scores between flushes, ie. the most work a kill -9 can cost


def shard_paths():
    """Every shard's token text, oldest first. Shards are enumerated from the directory rather than
    from a count, so a hand-deleted shard just drops out instead of shifting what the others mean."""
    return sorted(TOKEN_CACHE_DIR.glob('shard_*.tsv.gz'))


def sidecar_path(data_path):
    # the data shard holds the token text; the sidecar lists the names it contains, so a resume can
    # tell what is done without decompressing (and reparsing) gigabytes of shard
    return data_path.with_name(data_path.name.replace('.tsv.gz', '.names'))


def open_shard():
    """A fresh, empty shard, numbered past every shard already there."""
    used = [ int(re.search(r'\d+', p.name).group()) for p in shard_paths() ]
    data_path = TOKEN_CACHE_DIR / f'shard_{max(used, default=-1) + 1:03d}.tsv.gz'
    return (gzip.open(data_path, 'wt', encoding='utf-8', newline='\n', compresslevel=1),
            open(sidecar_path(data_path), 'w', encoding='utf-8', newline='\n'))


def cached_names():
    """Every score already parsed (or already known unparseable), by dir/file name."""
    names = set()
    for data_path in shard_paths():
        sidecar = sidecar_path(data_path)
        if sidecar.exists():
            # splitlines, not split: plenty of scores have spaces in their filenames
            names |= set(sidecar.read_text(encoding='utf-8').splitlines())
    return names


def tokenize_to_cache(files, workers=None):
    """Parse every score missing from the token cache and append its rows to the current shard.
    Scores are independent, so this is pure fan-out; the chunksize keeps the pool from paying a task
    round trip per file, and the big scores are spread out rather than bunched."""
    TOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    done = cached_names()
    # names carry the '!' prefix for failures, which is not part of any file's name; comparing the
    # stripped form means a score that failed to parse last run is not retried on this one
    todo = [ f for f in files if score_name(f) not in done and f'!{score_name(f)}' not in done ]
    print(f'Token cache holds {len(done)}/{len(files)} scores; tokenizing the remaining {len(todo)}')
    if not todo: return

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    written, pending = 0, []
    # a resumed run starts a fresh shard rather than reopening the last one, so a shard is only ever
    # written by the process that created it (its length is bookkeeping, not something to preserve)
    data, names = open_shard()
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            # map preserves input order, so zipping the results back onto files pairs each
            # sequence with the score it came from
            for file, (seq, err) in tqdm(zip(todo, pool.map(tokenize_file, todo, chunksize=8)), 'Tokenizing...', total=len(todo)):
                name = score_name(file)
                if err is None:
                    # written straight through instead of accumulated: 20k scores of token text is
                    # several GB of python strings, and holding them only to write them at the
                    # checkpoint would make the checkpoint interval a memory limit
                    data.write(f'#{name}\n' + ''.join('\t'.join(quint) + '\n' for quint in seq))
                pending.append(name if err is None else f'!{name}')
                written += 1
                if len(pending) >= CACHE_FLUSH_SCORES:
                    flush_shard(data, names, pending)
                if written >= CACHE_SHARD_SCORES:
                    flush_shard(data, names, pending)
                    data.close(); names.close()
                    data, names, written = *open_shard(), 0
    finally:
        flush_shard(data, names, pending)
        data.close(); names.close()
    print(f'Token cache now holds {len(cached_names())} scores')


def flush_shard(data, names, pending):
    # rows first, then the sidecar, never the other way round: a crash in between leaves rows that no
    # sidecar claims, which the next run simply parses again, whereas naming them first would mark
    # them done while their rows were still sitting in a buffer and lose them for good
    if not pending: return
    data.flush()
    names.write(''.join(f'{name}\n' for name in pending))
    names.flush()
    pending.clear()


def iter_token_cache():
    """Yields (name, [(tok, tok, tok, tok, tok), ...]) for every score in the cache, in shard order."""
    for data_path in shard_paths():
        sidecar = sidecar_path(data_path)
        if not sidecar.exists(): continue
        # only scores the sidecar claims are read back: a shard the killed run never closed ends in a
        # half-written score, and dropping it here is what keeps the cache's two halves in step. The
        # claimed names are exactly the ones whose rows were flushed before the sidecar named them,
        # and everything unclaimed is what the next run reparses.
        claimed = set(sidecar.read_text(encoding='utf-8').splitlines())
        name, seq = None, []
        try:
            with gzip.open(data_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    # a row cut off mid-write is the one row with no newline after it, and it can end
                    # mid-token ('VOI' for 'VOICE_0') without looking short, so stop rather than parse
                    if not line.endswith('\n'): break
                    if line.startswith('#'):
                        if name in claimed: yield name, seq
                        name, seq = line[1:-1], []
                    else:
                        seq.append(tuple(line[:-1].split('\t')))
        except (EOFError, OSError, zlib.error) as e:
            # the same unclosed shard, now hitting the missing gzip trailer
            print(f'{data_path.name}: truncated ({type(e).__name__}), keeping the scores read so far')
        if name in claimed: yield name, seq


# ---------------------------------------------------------------- stage 2: token cache -> dataset

def build_dataset_from_cache():
    """Vocab + data.npz from the token cache. Streams it twice rather than holding the corpus in
    memory: the shards are the copy of record, so nothing has to be kept between the passes."""
    counts, names = count_tokens(iter_token_cache())
    vocab = build_vocab_list(counts, plot=True)
    # only the original key is stored; the dataset transposes on load, so the vocab still has to
    # carry every shifted token (appending keeps the ids of everything already in it)
    for head in vocab:
        head.extend(sorted({ transpose_tok(tok, i) for tok in head for i in TRANSPOSE_INTERVALS } - set(head)))
    save_vocab_list(vocab)

    # convert to dataset. Arrays are saved under each score's own name (dir/file), NOT positionally:
    # the val split is persisted as a list of these names, so re-preprocessing with a different
    # corpus can never silently re-point the split at different scores the way indices would
    lookup = build_vocab_lookup(vocab)
    seen, stored, events = { name: 0 for name in names }, 0, 0
    tmp = DATASET_PATH.with_name(DATASET_PATH.name + '.tmp')
    # np.savez in all-arrays-at-once form would hold the whole encoded corpus (~8 GB on full PDMX)
    # alongside the file it is writing; the archive it produces is just uncompressed .npy members,
    # so writing them one at a time keeps only one score resident and leaves data.npz identical.
    # It lands under a temp name so an interrupted stage 2 cannot leave a half-written data.npz
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, seq in tqdm(iter_token_cache(), 'Encoding...', total=len(names)):
            seen[name] += 1
            # a name in the cache twice (two preprocess runs at once, say) would otherwise become two
            # archive members under one key, of which np.load would show whichever it happened to
            # index; keeping only the last copy is the same rule stage 1's resume follows
            if seen[name] < names[name]: continue
            # a score that tokenized to nothing would be saved as a (0,) array rather than (0,5),
            # which every consumer indexes as (N,5); it has no events to train on either way
            if not seq: continue
            ids = np.array(tokenize_text_to_ids(seq, vocab, lookup), dtype=np.int64)
            stored, events = stored + 1, events + len(ids)
            with archive.open(f'{name}.npy', 'w', force_zip64=True) as member:
                np.lib.format.write_array(member, ids, allow_pickle=False)
    os.replace(tmp, DATASET_PATH)
    print(f'Wrote {DATASET_PATH.name}: {stored} scores, {events} events, vocab heads {[ len(head) for head in vocab ]}')


def count_tokens(cache):
    """One pass over the cache: how often each token appears per field, and how many times each
    score name appears (so the encode pass can tell a superseded copy from the one to keep)."""
    counts = [{}, {}, {}, {}, {}]
    names = {}
    for name, seq in tqdm(cache, 'Counting tokens...'):
        names[name] = names.get(name, 0) + 1
        for quint in seq:
            for idx, tok in enumerate(quint):
                counts[idx][tok] = counts[idx].get(tok, 0) + 1
    return counts, names


def plot_vocab():
    with open(VOCAB_LIST_PLOT_PATH, 'r') as f:
        histogram = json.load(f)

    print(len(histogram))
    print(', '.join(histogram[0]))

    import matplotlib.pyplot as plt
    for idx, histo in enumerate(histogram):
        pairs = sorted(zip(histo.keys(), histo.values()), key=lambda x: x[0], reverse=True)
        keys, values = zip(*pairs)
        plt.bar(keys, values, color=['red', 'orange', 'lime', 'cyan', 'violet'][idx], edgecolor='black')
        plt.xticks(rotation=90, ha='right') 
        plt.title(f"HEAD {idx + 1}")
        plt.tight_layout()
        # plt.show()
    # axs[1,2].hist([ len(seq) for seq in token_sequences ], bins=200, edgecolor='black', color='yellow')


def load_dataset():
    """Returns (names, arrays): each score's dir/file name alongside its (N, 5) id array."""
    loaded = np.load(DATASET_PATH)
    names = list(loaded.files)
    return names, [ loaded[n] for n in names ]


def tokenize_text_to_ids(quint_tokens, vocab_list, lookup=None):
    # lookup is the vocab inverted once by the caller; .index() is a linear scan of the head per
    # field per token, which over the whole corpus costs more than the scan that produced the tokens
    if lookup is None: lookup = build_vocab_lookup(vocab_list)
    quint_ids = []
    for tok_seq in quint_tokens:
        quint_ids.append((lookup[0][tok_seq[0]], lookup[1][tok_seq[1]], lookup[2][tok_seq[2]], lookup[3][tok_seq[3]], lookup[4][tok_seq[4]]))
    return quint_ids


def build_vocab_lookup(vocab_list):
    return [ { tok: idx for idx, tok in enumerate(head) } for head in vocab_list ]


def transpose_tok(tok, interval):
    # only pitch moves under transposition: an absolute midi pitch just slides, while a key's tonic
    # and a chord's root are pitch classes that wrap inside the octave. Scale degrees are already
    # relative to the key, and nothing else in a row has anything to do with pitch.
    if not tok.startswith(('PITCH', 'KEY', 'ROOT')): return tok
    value = int(tok.split('_')[-1]) + interval
    return f"{tok.split('_')[0]}_{value if tok.startswith('PITCH') else value % 12}"


def detokenize_ids(quint_ids, vocab_list):
    quint_toks = []
    for tok_seq in quint_ids:
        quint_toks.append((vocab_list[0][tok_seq[0]], vocab_list[1][tok_seq[1]], vocab_list[2][tok_seq[2]], vocab_list[3][tok_seq[3]], vocab_list[4][tok_seq[4]]))
    return quint_toks


# Naming chords through music21 is ~95% of tokenization, and a score plays the same handful of
# voicings over and over, so every step that reaches into music21 is memoized on exactly the values
# it reads. The caches live for the process and only grow to the number of distinct chords in the
# corpus, which is a few thousand.
_root_cache = {}
_subset_cache = {}
_analyze_cache = {}


def _root_pitch_class(midis):
    if midis not in _root_cache:
        _root_cache[midis] = music21.chord.Chord(list(midis)).root().pitchClass
    return _root_cache[midis]


def _name_subset(midis, key_ctx, key_id):
    # name one candidate set of pitches as (degree, root, quality), or None when music21 can't spell
    # it as something CHORD_QUALITIES covers. Keyed on the pitches and the key, which is all it reads.
    ck = (midis, key_id)
    if ck in _subset_cache: return _subset_cache[ck]
    c = music21.chord.Chord(list(midis))
    # flip enharmonics (eg. F,G#,C -> F,Ab,C) to find the spelling with the shortest chord symbol
    c = min(
        (music21.chord.Chord([ p.getEnharmonic() if flip else p for p, flip in zip(c.pitches, flips) ]) for flips in itertools.product(*[ (False, True) if p.accidental else (False,) for p in c.pitches ])),
        key=lambda cand: len(music21.harmony.chordSymbolFigureFromChord(cand, includeChordType=True)[0])
    )
    # NOTE: chordSymbolFigureFromChord is not idempotent -- it leaves state on the chord that makes a
    # second call answer differently (eg. E2/A4/D5 names as 'EsusaddD,omitB' first and 'Cannot Be
    # Identified' after), so the winner having already been named inside the min() above silently
    # drops some chords here. Kept as-is on purpose: fixing it changes which chords the corpus gets.
    figure, kind = music21.harmony.chordSymbolFigureFromChord(c, includeChordType=True)
    result = None
    if kind:
        quality = re.split(r'/|add|omit|dim5', figure[len(c.root().name):])[0]
        if quality in CHORD_QUALITIES:
            rn = music21.roman.romanNumeralFromChord(c, key_ctx)
            base = (key_ctx.pitchFromDegree(rn.scaleDegree).pitchClass - key_ctx.tonic.pitchClass) % 12
            alter = int(rn.frontAlterationAccidental.alter) if rn.frontAlterationAccidental else 0
            result = (base + alter) % 12, c.root().midi % 12, quality
    _subset_cache[ck] = result
    return result


def analyze_chord(weights, voicing, key_ctx, key_id=None, key_pcs=None):
    # name a window's pitch classes (weights = how long each sounds, voicing = the lowest pitch it
    # sounds at, kept so the real bass still drives root detection) as (degree, root, quality), or
    # None if they make no chord we have a token for. Only the heaviest tone of each slot above the
    # root survives, so 9ths/11ths/13ths are stripped and a 7th is the richest possible result, and
    # whatever the slots can't settle (eg. E and F over C) falls to whichever sounds longer.
    # key_id/key_pcs are the caller's cached read of key_ctx; derived here when called standalone.
    if key_id is None: key_id = (key_ctx.tonic.name, key_ctx.mode)
    if key_pcs is None: key_pcs = { p.pitchClass for p in key_ctx.pitches }
    # longest sounding tone claims each slot, and the lower voice wins an outright tie
    order = tuple(sorted(weights, key=lambda pc: (-weights[pc], voicing[pc])))
    ck = (order, tuple(sorted(voicing.items())), key_id)
    if ck in _analyze_cache: return _analyze_cache[ck]

    root = _root_pitch_class(tuple(sorted(voicing.values())))
    kept = {}
    for pc in order:
        kept.setdefault(CHORD_SLOTS.get((pc - root) % 12, -1), pc)
    kept.pop(-1, None)
    # two lone tones spell a chord only when they form a real third; a second or a fourth between
    # them is a melodic collision rather than a suspension, and a lone root+seventh is nothing
    if len(kept) == 2 and (kept.get(1, root) - root) % 12 not in (3, 4):
        _analyze_cache[ck] = None
        return None
    # music21 can't name a chord that has no fifth, but an omitted fifth doesn't change what the
    # chord is, so stand one in, taking its alteration from the key so a leading tone gets the
    # diminished fifth it wants (B in C major -> F, not F#); a real fifth would fill the slot itself
    if 2 not in kept:
        fifth = next((f for f in (7, 6, 8) if (root + f) % 12 in key_pcs), 7)
        kept[2] = (root + fifth) % 12
        voicing[kept[2]] = voicing[root] + fifth
    # the 7th chord first, then the bare triad if that wasn't nameable
    result = None
    for subset in dict.fromkeys([ tuple(sorted(kept)), tuple(slot for slot in sorted(kept) if slot != 3) ]):
        if len(subset) < 3: continue
        named = _name_subset(tuple(sorted(voicing[kept[slot]] for slot in subset)), key_ctx, key_id)
        if named is not None:
            result = named
            break
    _analyze_cache[ck] = result
    return result


def chord_walker(chordified_measure, key_ctx):
    # walk the measure in ever wider windows (half quarter -> quarter -> half -> whole bar),
    # analyzing a window only when nothing finer inside it already resolved, so wide spans just
    # pool the leftovers of a slow harmonic rhythm instead of blurring chords that were found
    bar_ql = float(chordified_measure.barDuration.quarterLength)
    # read the key once for the whole measure instead of once per window
    key_id, key_pcs = (key_ctx.tonic.name, key_ctx.mode), { p.pitchClass for p in key_ctx.pitches }
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
            named = analyze_chord(weights, voicing, key_ctx, key_id, key_pcs)
            if named: chords[pos] = (named, step)
    return [ (pos, *chords[pos][0]) for pos in sorted(chords) ]


def canonical_instrument(instr):
    # the same instrument reaches us under every name a typesetter felt like ("Grand Piano",
    # "Piano (2)", "Klavier", "Hautbois"), which splits it across that many vocab entries. Run the
    # name through music21's own multilingual lookup and keep the plain name of whatever class it
    # matched; parenthesized asides, digits and punctuation are stripped first so they can't outvote
    # the real name (eg. "Horn (bass clef)" matching "bass", "Piano (2)" not matching at all). When
    # the name means nothing to music21 ("Men", "Traverso"), the class it already resolved from the
    # part's midi program still says what the instrument is; when even that is the generic
    # Instrument base class (whose default name is None), fall back to Piano rather than letting the
    # raw typesetter string ("bass2 ou instr.", "(10)") become its own vocab entry.
    name = re.sub(r'\(.*?\)', '', instr.instrumentName or '')
    name = re.sub(r'[\d()\[\].,;:!?]+', ' ', name).strip()
    try:
        matched = music21.instrument.fromString(name) if name else instr
    except music21.instrument.InstrumentException:
        matched = instr
    return type(matched)().instrumentName or 'Piano'


def tokenize_score_as_text(score: music21.stream.Score):
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
    # construct instruments; parts past MAX_VOICES merge into the last voice (a 69-slot voice head
    # is a long tail of orchestral scores that mostly steals capacity from chord qualities)
    instr_toks = []
    for idx, part in enumerate(score.parts[:MAX_VOICES]):
        name = canonical_instrument(part.getInstrument())
        instr_toks.append((tokify('INSTR', name), '', '', '', tokify('VOICE', idx)))

    chordified = score.chordify()
    score_toks = []
    seen_bars = set()
    last_chord = (None, None)
    # no progress bars in here: this runs inside a worker pool, where per-measure bars from every
    # process interleave into noise and the writes themselves cost more than the work they track
    for voz_idx, part in enumerate(score.parts):
        voz = min(voz_idx, MAX_VOICES - 1)
        measure_offset = 0
        for measure in part.getElementsByClass(music21.stream.Measure):
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

            # read all notes and convert to text toks. Offsets/durations are rounded to the
            # subdivision grid rather than left fractional (quintuplets etc. produce POS_4.8 and
            # DUR_1.5 tokens the decoder can't turn back into numbers), negative offsets are
            # music21 parse artifacts (the corpus holds a POS_-4026), and pitches outside the
            # piano keep ~30 never-musical embedding rows out of the vocab
            note_toks = []
            for note in notesAndRests:
                if note.isRest:
                    continue

                pos = round(note.offset * SUBDIVISIONS_PER_QUARTER)
                dur = round(note.duration.quarterLength * SUBDIVISIONS_PER_QUARTER)
                if dur == 0 or pos < 0: continue

                for pitch in note.pitches:
                    if not (MIN_PITCH <= pitch.midi <= MAX_PITCH): continue
                    note_toks.append((tokify('M', m_num), tokify('POS', pos), tokify('DUR', dur), tokify('PITCH', pitch.midi), tokify('VOICE', voz)))

            # read all key sigs
            key_toks = []
            if voz_idx == 0:
                for key_sig in measure.getElementsByClass(music21.key.KeySignature):
                    tonic = key_sig.asKey().tonic.midi % 12
                    pos = round(key_sig.offset * SUBDIVISIONS_PER_QUARTER)
                    if pos < 0: continue
                    key_toks.append((tokify('M', m_num), tokify('POS', pos), tokify('KEY', tonic), '', ''))

            # read all time sigs
            time_toks = []
            if voz_idx == 0:
                for ts in measure.getElementsByClass(music21.meter.TimeSignature):
                    pos = round(ts.offset * SUBDIVISIONS_PER_QUARTER)
                    if pos < 0: continue
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
                    # clamp: the corpus carries TEMPO_75000 (1.2M BPM) and TEMPO_0 typesetter garbage
                    if mm.number is None or not (MIN_TEMPO <= mm.number <= MAX_TEMPO): continue
                    pos = round(mm.offset * SUBDIVISIONS_PER_QUARTER)
                    if pos < 0: continue
                    tempo_toks.append((tokify('M', m_num), tokify('POS', pos), tokify('TEMPO', round(mm.number / 16)), '', ''))

            # read all dynamics
            dynamic_toks = []
            for dyn in measure.getElementsByClass(music21.dynamics.Dynamic):
                pos = round(dyn.offset * SUBDIVISIONS_PER_QUARTER)
                if pos < 0: continue
                dynamic_toks.append((tokify('M', m_num), tokify('POS', pos), tokify('DYN', dyn.value), '', tokify('VOICE', voz)))

            # read all clefs
            clef_toks = []
            for clef in measure.getElementsByClass(music21.clef.Clef):
                pos = round(clef.offset * SUBDIVISIONS_PER_QUARTER)
                if pos < 0: continue
                clef_toks.append((tokify('M', m_num), tokify('POS', pos), tokify('CLEF', clef.name), '', tokify('VOICE', voz)))

            # read all bar lines (including repeats); not always on the first staff, so check every part
            bar_toks = []
            for barline in measure.getElementsByClass(music21.bar.Barline):
                pos = round(barline.offset * SUBDIVISIONS_PER_QUARTER)
                if pos < 0: continue
                if (m_num, pos) in seen_bars: continue
                seen_bars.add((m_num, pos))
                bar_type = f'repeat-{barline.direction}' if isinstance(barline, music21.bar.Repeat) else barline.type
                bar_toks.append((tokify('M', m_num), tokify('POS', pos), tokify('BAR', bar_type), '', ''))

            # walk the measure for chords, analyzed across all staves; skip repeats of the same chord
            chord_toks = []
            if voz_idx == 0:
                chordified_measure = chordified.measure(m_num)
                if chordified_measure is not None:
                    key_ctx = (measure.getContextByClass(music21.key.KeySignature) or music21.key.KeySignature(0)).asKey()
                    for pos, degree, root, quality in chord_walker(chordified_measure, key_ctx):
                        if (root, quality) == last_chord: continue
                        last_chord = (root, quality)
                        chord_toks.append((tokify('M', m_num), tokify('POS', round(pos * SUBDIVISIONS_PER_QUARTER)), tokify('DEG', degree), tokify('ROOT', root), tokify('QUAL', quality, lower=False)))

            # extend all to score_toks
            score_toks.extend(note_toks)
            score_toks.extend(key_toks)
            score_toks.extend(time_toks)
            score_toks.extend(tempo_toks)
            score_toks.extend(dynamic_toks)
            score_toks.extend(clef_toks)
            score_toks.extend(bar_toks)
            score_toks.extend(chord_toks)

    # final sort
    score_toks.sort(key=lambda x: tok_priority(x))

    # remap absolute bar numbers to relative advance tokens. The sort above needs absolute bars,
    # but as vocab entries M_0..M_2270 are a ~2.3k-way head of near-uniform counters the model can
    # only memorize (99.5% of field-0 mass), and no score can outgrow the largest bar index seen in
    # training. "Does time advance?" is the musical content, so keep only that: M_SAME within a
    # bar, M_NEXT into the next, M_SKIP_<k> (clamped) over a gap. tokens_to_score reverses this
    # with a running counter; bars are non-decreasing here by construction of tok_priority.
    rel_toks = []
    prev_bar = None
    for tok in score_toks:
        bar = int(tok[0].split('_')[-1])
        if prev_bar is None or bar == prev_bar:
            m_tok = 'M_SAME'
        elif bar - prev_bar == 1:
            m_tok = 'M_NEXT'
        else:
            m_tok = f'M_SKIP_{min(bar - prev_bar, MAX_BAR_SKIP)}'
        prev_bar = bar
        rel_toks.append((m_tok, *tok[1:]))
    score_toks = rel_toks

    # constructed
    output = [
        ('SOG', '', '', '',''),
        *instr_toks,
        ('SOS', '', '', '',''),
        *score_toks,
        ('EOS', '', '', '', '')
    ]

    return output


def dump_tokens(tokens, path=None):
    # eyeball a single score's tokens; used to be written on every tokenize, which cost a full
    # str()+join of the score per file and made the whole pass unsafe to run in parallel
    with open(path or STORAGE_DIR / 'urmom.txt', 'w') as f:
        f.writelines('\n'.join([ str(t) for t in tokens ]))


def pretty(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print('  ' * indent + str(key))
            pretty(value, indent+1)
        else:
            print('  ' * (indent+1) + str(key) + str(value))


def resolve_bars(tokens):
    # invert the relative-bar remap in tokenize_score_as_text: walk the rows with a running
    # absolute counter so every downstream consumer keeps seeing plain M_<n>. Legacy absolute
    # tokens pass through untouched, so pre-remap token dumps still decode.
    resolved = []
    bar = 0
    for tok in tokens:
        f0 = tok[0]
        if f0.startswith('M_'):
            tail = f0[2:]
            if tail == 'SAME':
                pass
            elif tail == 'NEXT':
                bar += 1
            elif tail.startswith('SKIP_'):
                bar += int(tail.split('_')[-1])
            else:
                bar = int(tail)
            resolved.append((f'M_{bar}', *tok[1:]))
        else:
            resolved.append(tok)
    return resolved


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

    tokens = resolve_bars(tokens)
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
    score.write("musicxml", STORAGE_DIR / "output.mxl", makeNotation=True)


def build_vocab_list(counts, plot=False):
    """One head per field, ids in order of first appearance in the corpus. `counts` is count_tokens'
    per-field token -> occurrences, which is already keyed by every token there is."""
    # the frequencies are only ever plotted, and plotting cannot happen during a headless remote
    # preprocess run, so they are saved for plot_vocab() to draw later instead of drawn here
    if plot:
        with open(VOCAB_LIST_PLOT_PATH, 'w') as f:
            json.dump(counts, f)
    return [ list(head) for head in counts ]


def save_vocab_list(vocab):
    with open(VOCAB_LIST_PATH, 'w') as f:
        json.dump(vocab, f)


def load_vocab_list():
    with open(VOCAB_LIST_PATH, 'r') as f:
        return json.load(f)


class QuintupleTokenizer:
    def __init__(self):
        vocab_list = load_vocab_list()
        vocab_dict = [ { tok: idx for idx, tok in enumerate(head) } for head in vocab_list ]
        self.vocabs = vocab_dict
        self.field_sizes = [len(v) for v in self.vocabs]
        self.pad_ids = [None] + [self.vocabs[i][''] for i in range(1, 5)]
        self.inv = [{i: s for s, i in v.items()} for v in self.vocabs]
        # structural-consistency sets for constrained sampling (generate.py): rows whose field 0
        # is structural carry '' in fields 1-3 and either '' (SOG/SOS/EOS) or a VOICE (INSTR) in
        # field 4; everything the model samples outside that is a malformed row by construction
        self.struct_f0 = { idx for tok, idx in self.vocabs[0].items() if tok in ('SOG', 'SOS', 'EOS') or tok.startswith('INSTR') }
        self.instr_f0 = { idx for tok, idx in self.vocabs[0].items() if tok.startswith('INSTR') }
        self.voice_f4 = { idx for tok, idx in self.vocabs[4].items() if tok.startswith('VOICE') }

    def encode(self, quint_tokens):
        quint_ids = []
        for tok_seq in quint_tokens:
            quint_ids.append((self.vocabs[0][tok_seq[0]], self.vocabs[1][tok_seq[1]], self.vocabs[2][tok_seq[2]], self.vocabs[3][tok_seq[3]], self.vocabs[4][tok_seq[4]]))
        return quint_ids

    def decode(self, quint_ids):
        quint_toks = []
        for tok_seq in quint_ids:
            quint_toks.append((self.inv[0][tok_seq[0]], self.inv[1][tok_seq[1]], self.inv[2][tok_seq[2]], self.inv[3][tok_seq[3]], self.inv[4][tok_seq[4]]))
        return quint_toks


if __name__  == "__main__":
    # preprocess_dataset()

    # # validate
    # rng = np.random.default_rng()
    # quint_id_seqs = load_dataset()
    # vocab_list = load_vocab_list()
    # detokenized = [ detokenize_ids(seq, vocab_list) for seq in quint_id_seqs ]
    # chosen = detokenized[rng.choice(len(detokenized))]
    # tokens_to_score(chosen)
    plot_vocab()
