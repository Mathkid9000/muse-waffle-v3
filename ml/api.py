"""Everything the FastAPI layer needs from the model, in the model's own terms.

The three creative endpoints are one operation: build a prefix of quintuple rows, let the model
sample rows on from there, and re-render the score every time a bar closes so the frontend can
stream a measure at a time. Generating from scratch seeds the prefix with a header, appending
uses the whole existing score as the prefix, and infilling cuts the prefix at the gap and splices
the untouched tail back on afterwards.

Nothing in here reaches into the model, the vocabulary or the token layout -- it drives
generate_score and tokens_to_score exactly as ml/generate.py's own __main__ does -- so the API
keeps working across a retokenize or a retrain without knowing either happened.
"""

from __future__ import annotations
import logging, threading
import music21
from generate import generate_score
from helpers import STORAGE_DIR
from infill import canvas_to_tokens, infill, repair_seams
from model_util import load_model_only
from tokenizer import (MAX_BAR_SKIP, MAX_VOICES, QuintupleTokenizer, canonical_instrument,
                       resolve_bars, tokenize_score_as_text, tokens_to_score)

DEFAULT_MODEL = 'the-waffler'
FILL_MODEL = 'the-waffiller'   # the masked model: it writes a canvas's holes rather than a continuation
TIME_SIGS = ('2/4', '3/4', '4/4', '6/8', '3/8', '12/8')   # the API's time-signature indices

# Indexed to match the frontend's instrument picker. The names are only a starting point: each one
# is run through the same canonicalization a real score's part name gets, so it lands on the token
# the corpus actually uses rather than on a spelling the model has never seen.
INSTR_NAMES = [
    'Piano', 'Piano', 'Harpsichord', 'Organ', 'Accordion', 'Guitar', 'Bass Guitar',
    'Violin', 'Viola', 'Cello', 'Double Bass', 'Harp', 'Flute', 'Oboe',
    'Clarinet', 'Bassoon', 'Saxophone', 'Horn', 'Trumpet', 'Trombone', 'Tuba',
    'Percussion', 'Timpani', 'Celesta', 'Soprano', 'Mezzo-Soprano', 'Alto',
    'Tenor', 'Baritone', 'Bass', 'Voice', 'Choir',
]

_loaded: dict[tuple[str, str], tuple] = {}
_render_lock = threading.Lock()


def _load(model_name, device):
    """Model, config and tokenizer for a checkpoint, kept per (checkpoint, device): a request
    would otherwise reload tens of millions of parameters, and the whole vocabulary with them."""
    if (model_name, device) not in _loaded:
        model, cfg = load_model_only(model_name, device)
        model.eval()
        _loaded[(model_name, device)] = (model, cfg, QuintupleTokenizer())
    return _loaded[(model_name, device)]


# ---------------------------------------------------------------- rows

def _voice(row):
    """The staff a row belongs to, or None for the rows that belong to the score rather than to
    any one staff (key, time, tempo, barlines, chord symbols)."""
    return int(row[4].split('_')[-1]) if row[4].startswith('VOICE') else None


def _rewritten(row, targets):
    """Whether a row in a rewritten block comes from the new material rather than the old score:
    every row when all staves are rewritten, otherwise the target staves' rows plus the chord
    symbols, which are read off every staff's notes and so change whenever any of them does."""
    return targets is None or _voice(row) in targets or row[2].startswith('DEG')


def _bar_after(bar, f0):
    """The absolute bar a row lands in, given the bar the row before it landed in. This is
    resolve_bars' rule one row at a time, which is what streaming needs: the bar has to be known
    as each row is sampled, not once the sequence is finished."""
    if not f0.startswith('M_'): return bar
    tail = f0[2:]
    if tail == 'SAME': return bar
    if tail == 'NEXT': return bar + 1
    if tail.startswith('SKIP_'): return bar + int(tail.split('_')[-1])
    return int(tail)


def _last_bar(rows):
    """The bar the last of `rows` sits in, ie. the bar a continuation carries on from."""
    bar = 0
    for row in rows:
        bar = _bar_after(bar, row[0])
    return bar


def _score_rows(xml_str):
    """An existing score as quintuple rows, EOS dropped so the model reads it as unfinished."""
    score = music21.converter.parseData(xml_str, format='musicxml')
    # the engraver writes a note that crosses a beat or a barline as tied pieces; read back as one
    # event, so the next render sees the note that was played, not the pieces it was written as
    score.stripTies(inPlace=True)
    return [ row for row in tokenize_score_as_text(score) if row[0] != 'EOS' ]


def _encode(tokenizer, rows):
    try:
        return tokenizer.encode(rows)
    except KeyError as e:
        raise RuntimeError(
            f'token {e.args[0]!r} is not in the vocabulary. vocab.json and the checkpoint have to '
            f'come from the same preprocessing run -- retrain, or point model_name at a checkpoint '
            f'built against this vocabulary.') from e


# ---------------------------------------------------------------- rendering

def _well_formed(row):
    """Whether a row is typed the way the decoder reads it.

    Fields are sampled one at a time, so nothing stops the model from pairing a time signature
    with no denominator or a pitch with no duration. It is rare, and it is fatal: the decoder
    reads the companion field as a number and there isn't one. Dropping the row costs one event,
    letting it through costs the whole snapshot.
    """
    if row[3].startswith('PITCH'): return row[2].startswith('DUR')
    if row[2].startswith('NUM'): return row[3].startswith('DEN')
    if row[2].startswith('DEG'): return row[3].startswith('ROOT') and row[4].startswith('QUAL')
    return True


def _to_xml(rows):
    """MusicXML for a snapshot of rows, or None while it holds nothing renderable yet.

    Only the staves that actually have notes are rendered: tokens_to_score pairs the Nth INSTR row
    with the notes of voice N, so a staff the model hasn't played yet -- which early in a stream is
    most of them -- would otherwise take the whole snapshot down. The played voices are renumbered
    onto 0..n-1 and the headers of the silent ones dropped, so a snapshot that happens to hold only
    voice 1 renders as that one staff instead of as nothing at all. Staves come back, in their own
    order, as soon as they get a note.
    """
    rows = [ row for row in rows if _well_formed(row) ]
    voiced = sorted({ _voice(row) for row in rows if row[3].startswith('PITCH') } - { None })
    if not voiced:
        return None
    seat = { voice: staff for staff, voice in enumerate(voiced) }

    trimmed, headers = [], 0
    for row in rows:
        if row[0].startswith('INSTR'):
            # headers need not arrive in voice order, so they are matched to voices by position
            headers += 1
            if headers - 1 not in seat: continue
        elif _voice(row) is not None:
            if _voice(row) not in seat: continue
            row = (*row[:4], f'VOICE_{seat[_voice(row)]}')
        trimmed.append(row)
    rows = trimmed

    # tokens_to_score also writes the score out as a debugging aid, so the lock keeps two streams
    # from interleaving their writes to that file
    with _render_lock:
        try:
            return tokens_to_score(rows)
        except Exception:
            # a snapshot the decoder chokes on costs one frame of the stream rather than the whole
            # run, but it is never nothing: say what broke and leave the rows behind to look at
            (STORAGE_DIR / 'failed_rows.txt').write_text('\n'.join(map(str, rows)), encoding='utf-8')
            logging.getLogger(__name__).error(
                'Could not render snapshot of %d rows (dumped to storage/failed_rows.txt)',
                len(rows), exc_info=True)
            return None


# ---------------------------------------------------------------- sampling

def _stream_bars(model, tokenizer, prefix, first_new_bar, num_bars, max_new,
                 temperature, top_k, seq_len):
    """Sample rows on from `prefix`, yielding (bars finished, rows so far) every time a bar closes.

    A bar is only handed over once it is finished -- the bar being sampled is half-written by
    definition, and rendering it would flicker notes into a measure that is still filling up.
    Rows come back tagged with the absolute bar they landed in, which is what splicing needs.

    Sampling runs a row at a time so the caller can stop at `num_bars` (the model has no notion of
    a target length), and EOS stays blocked throughout: the prefix of an append or an infill is
    exactly the point where a piece would have ended, so left to itself the model would close it.
    """
    ids = _encode(tokenizer, prefix)
    bar, rows = _last_bar(prefix), []
    for _ in range(max_new):
        ids = generate_score(model, tokenizer, ids, max_events=len(ids) + 1, seq_len=seq_len,
                             temperature=temperature, top_k=top_k, min_events=len(ids) + 1)
        row = tuple(tokenizer.inv[field][int(tok)] for field, tok in enumerate(ids[-1]))
        landed = _bar_after(bar, row[0])
        if landed > bar:
            done = bar - first_new_bar + 1
            if done > 0:
                yield done, rows
                if done >= num_bars: return
            bar = landed
        rows.append((bar, row))
    yield max(bar - first_new_bar + 1, 0), rows


# ---------------------------------------------------------------- endpoints

def _instr_token(tokenizer, instr):
    """The INSTR token for a picker index. Anything the corpus never named falls back to piano: an
    unknown token has no embedding row, and a header the model can't read is worse than the wrong
    instrument."""
    name = INSTR_NAMES[instr] if 0 <= instr < len(INSTR_NAMES) else 'Piano'
    canonical = canonical_instrument(music21.instrument.Instrument(instrumentName=name))
    tok = f'INSTR_{canonical.lower()}'
    return tok if tok in tokenizer.vocabs[0] else 'INSTR_piano'


def _header_rows(key, time_sig, tempo, first='M_SAME'):
    """The key, time signature and tempo a generated block opens with, `first` being the bar advance
    that lands them in its first bar. A field left as None emits nothing at all, so it carries on
    from whatever the prefix already put in effect."""
    fields = []
    if key is not None:
        tonic = (7 * (key - 6)) % 12   # key index 0..12 is 6 flats..6 sharps, ie. a fifth per step
        fields.append((f'KEY_{tonic}', ''))
    if time_sig is not None:
        num, den = TIME_SIGS[time_sig].split('/')
        fields.append((f'NUM_{num}', f'DEN_{den}'))
    if tempo is not None:
        fields.append((f'TEMPO_{tempo}', ''))
    return [ (first if i == 0 else 'M_SAME', 'POS_0', a, b, '') for i, (a, b) in enumerate(fields) ]


def _advance(delta):
    """The bar token moving `delta` bars on from wherever the prefix left off."""
    if delta <= 0: return 'M_SAME'
    if delta == 1: return 'M_NEXT'
    return f'M_SKIP_{min(delta, MAX_BAR_SKIP)}'


def _generate_seed(tokenizer, instruments, key, time_sig, tempo):
    """The rows a score opens with: SOG, one INSTR row per staff, SOS, then the first bar's key,
    time signature and tempo. Seeding the header instead of sampling it keeps the piece on the
    instrumentation that was asked for -- sampled at temperature it wanders through combinations,
    and a run that never emits an INSTR row can't be decoded into a score at all."""
    output = [('SOG', '', '', '', '')]
    if instruments:
        output.extend([ (_instr_token(tokenizer, instr), '', '', '', f'VOICE_{staff}')
           for staff, instr in enumerate(instruments[:MAX_VOICES]) ])
        output.append(('SOS', '', '', '', ''))
        output.extend(_header_rows(key, time_sig, tempo))

    return output


def generate_stream(xml_str, start_measure, instruments, parts, key, time_sig, tempo, num_bars,
                    overwrite=False, max_new=2000, temperature=0.9, top_k=50, device='cpu',
                    model_name=DEFAULT_MODEL, log=print):
    """Stream `num_bars` measures generated at `start_measure`, counted from 1 as the frontend
    renders them.

    Yields (bars finished, MusicXML, bars asked for, staves) once per finished bar. With no score to
    work from the whole piece is sampled off a seeded header; otherwise everything before the block
    is the prefix the model reads, so the generation continues the music instead of starting blind.
    `overwrite` replaces the measures already sitting in the block; without it those measures are
    pushed down by `num_bars` and the block is inserted in front of them, keeping their own contents.
    `parts` narrows the generation to those staves and leaves the rest of the block as it was, the
    way infill_stream does.
    """
    model, cfg, tokenizer = _load(model_name, device)
    rows = _score_rows(xml_str) if xml_str else []
    start = max(start_measure - 1, 0) if rows else 0
    end = start + num_bars - 1
    # a score built from scratch has no staves to leave alone, so the staff list is instrumentation
    # rather than a filter -- filtering there would only drop the rows that belong to no staff
    targets = set(parts) if parts and rows else None

    if not rows:
        prefix, kept, tail = _generate_seed(tokenizer, instruments, key, time_sig, tempo), [], []
        if not instruments:
            # gen instruments
            max_instr = 20
            encoded_prefix = _encode(tokenizer, prefix)
            prefix_ended = False
            print(prefix)
            for _ in range(max_instr):
                encoded_prefix = generate_score(model, tokenizer, encoded_prefix, max_events=len(encoded_prefix) + 1, seq_len=max_instr,
                    temperature=temperature, top_k=top_k, min_events=len(encoded_prefix) + 1)
                row = tuple(tokenizer.inv[field][int(tok)] for field, tok in enumerate(encoded_prefix[-1]))
                prefix.append(row)
                print(prefix)
                if row[0] == 'SOS':
                    prefix_ended = True
                    break
            if not prefix_ended:
                print('Error generating prefix: ', prefix)
                return
    else:
        # the block is cut in resolved-bar space: the relative rows say how far time moves, not
        # where it is, and the frontend's Nth measure is the Nth bar of the stream either way
        absolute = resolve_bars(rows)
        bars = [ int(row[0][2:]) if row[0].startswith('M_') else -1 for row in absolute ]

        # the prefix stays relative (it is what the model reads) and keeps the SOG/INSTR/SOS header,
        # which carries no bar; everything spliced back in is pinned to its own absolute bar, so it
        # cannot drift with however many rows the block ends up filling with
        prefix = [ row for row, bar in zip(rows, bars) if bar < start ]
        if overwrite:
            kept = [ row for row, bar in zip(absolute, bars) if start <= bar <= end
                     and not _rewritten(row, targets) ]
            tail = [ row for row, bar in zip(absolute, bars) if bar > end ]
        else:
            kept = []
            tail = [ (f'M_{bar + num_bars}', *row[1:])
                     for row, bar in zip(absolute, bars) if bar >= start ]
        prefix = [ *prefix, *_header_rows(key, time_sig, tempo, _advance(start - _last_bar(prefix))) ]

    staves = sum(1 for row in prefix if row[0].startswith('INSTR'))
    log(f'Generating {num_bars} bars from bar {start} for {staves} staves with {model_name}')

    rendered = 0
    for done, rows_out in _stream_bars(model, tokenizer, prefix, start, num_bars, max_new,
                                       temperature, top_k, cfg['max_seq_len']):
        block = [ (f'M_{bar}', *row[1:]) for bar, row in rows_out
                  if bar >= start and _rewritten(row, targets) ]
        xml = _to_xml([ *prefix, *block, *kept, *tail ])
        if xml is None: continue
        rendered += 1
        log(f'Generated bar {done} of {num_bars}')
        yield done, xml, num_bars, staves

    # a run where nothing was renderable used to end like a successful one, leaving the caller with
    # the score it started with and no reason why
    if not rendered:
        raise RuntimeError('Generation produced no renderable bar — try again, or ask for more bars')


def append_stream(xml_str, staves, num_bars=4, max_new=1024, temperature=0.9, top_k=50,
                  device='cpu', model_name=DEFAULT_MODEL, log=print):
    """Stream `num_bars` new measures onto the end of an existing score.

    Yields (new bars finished, MusicXML, bars asked for) once per finished bar. The score comes
    back re-rendered from its own tokens rather than edited in place, so what the frontend gets is
    the tokenizer's reading of the original with the new measures on the end.
    """
    model, cfg, tokenizer = _load(model_name, device)
    prefix = _score_rows(xml_str)
    log(f'Appending {num_bars} bars after bar {_last_bar(prefix)} with {model_name}')

    for done, rows in _stream_bars(model, tokenizer, prefix, _last_bar(prefix) + 1, num_bars,
                                   max_new, temperature, top_k, cfg['max_seq_len']):
        xml = _to_xml([ *prefix, *(row for _, row in rows) ])
        if xml is None: continue
        log(f'Appended bar {done} of {num_bars}')
        yield done, xml, num_bars


def infill_stream(xml_str, start_measure, end_measure, parts=None, max_new=1500,
                  temperature=0.9, top_k=50, device='cpu', model_name=DEFAULT_MODEL, log=print):
    """Stream a resampling of measures [start_measure, end_measure], counted from 1 as the
    frontend renders them.

    Yields (gap bars finished, MusicXML, gap length) once per finished bar. The model only reads
    left to right, so the gap is conditioned on the music before it; the measures after it are
    spliced back on unchanged, keeping their own bar numbers, so the rest of the score doesn't
    move while the gap streams. With `parts` given, only those staves are resampled and the rest
    of the gap -- the other staves, plus the key, time, tempo and barlines the whole score
    shares -- is kept as it was, as vertical context for the ear if not for the model. Chord
    symbols are the exception: they are the harmony of every staff, so the model's replace them.
    """
    model, cfg, tokenizer = _load(model_name, device)
    rows = _score_rows(xml_str)
    # the gap is cut in resolved-bar space: the relative rows say how far time moves, not where it
    # is, and the frontend's Nth measure is the Nth bar of the stream either way
    absolute = resolve_bars(rows)
    bars = [ int(row[0][2:]) if row[0].startswith('M_') else -1 for row in absolute ]
    start, end = start_measure - 1, end_measure - 1
    targets = set(parts) if parts is not None else None

    # the prefix stays relative (it is what the model reads); everything spliced back in is pinned
    # to its own absolute bar, so it cannot drift if the gap fills with a different number of rows
    prefix = [ row for row, bar in zip(rows, bars) if bar < start ]
    kept = [ row for row, bar in zip(absolute, bars)
             if start <= bar <= end and not _rewritten(row, targets) ]
    tail = [ row for row, bar in zip(absolute, bars) if bar > end ]
    log(f'Filling bars {start_measure}-{end_measure} of {max(bars) + 1} with {model_name}')

    for done, rows_out in _stream_bars(model, tokenizer, prefix, start, end - start + 1, max_new,
                                       temperature, top_k, cfg['max_seq_len']):
        filled = [ (f'M_{bar}', *row[1:]) for bar, row in rows_out
                   if bar >= start and _rewritten(row, targets) ]
        xml = _to_xml([ *prefix, *filled, *kept, *tail ])
        if xml is None: continue
        log(f'Filled bar {done} of {end - start + 1}')
        yield done, xml, end - start + 1


def fill_template(rows, steps=12, temperature=1.0, device='cpu', model_name=FILL_MODEL, log=print):
    """MusicXML for a canvas of rows, every MASK in it written by the masked model.

    This is ml/infill.py's own __main__ as an endpoint: fill the canvas, rewrite the rows either
    side of the ones the caller gave now that there is music around them, and render what is left
    once the slack has been dropped. A field the vocabulary doesn't hold is handed over as a MASK
    rather than refused -- a template can name a chord quality or an instrument the corpus never
    did, and letting the model choose that one field is nearer what was asked for than a failure.
    """
    model, cfg, tokenizer = _load(model_name, device)
    if len(rows) > cfg['max_seq_len']:
        raise RuntimeError(f'Template fill failed: {len(rows)} rows is past the model\'s '
                           f'{cfg["max_seq_len"]}-row canvas; ask for fewer bars or slots')
    rows = [ tuple(tok if not tok or tok in tokenizer.vocabs[i] else 'MASK' for i, tok in enumerate(row))
             for row in rows ]
    log(f'Filling {sum(row.count("MASK") for row in rows)} slots of a {len(rows)}-row canvas with {model_name}')

    filled, order = infill(model, tokenizer, rows, steps=steps, temperature=temperature)
    repaired, order = repair_seams(model, tokenizer, rows, filled, order)
    tokens, _ = canvas_to_tokens(tokenizer, filled, order)
    xml = _to_xml(tokens)
    if xml is None:
        raise RuntimeError('Template fill failed: the model wrote no notes into the template')
    return xml
