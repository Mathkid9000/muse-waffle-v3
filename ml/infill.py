import math
import torch
import torch.nn.functional as F

from model_util import load_model_only
from tokenizer import QuintupleTokenizer, resolve_bars, tokens_to_score

MASKED = ('MASK', 'MASK', 'MASK', 'MASK')


def masked_bars(bars, slots_per_bar, start_next=True):
    """The canvas a caller hands over for a region it wants written: one bar advance per bar,
    the rest of that bar's slots left at the same bar, every other field the model's to choose.

    Over-allocate ``slots_per_bar`` — the model empties the slack into NULL rows, but it cannot
    add slots it was never given, so a tight budget caps the region's event density."""
    rows = []
    for i in range(bars):
        if i != 0 or start_next:
            rows.append(('M_NEXT', *MASKED))
        rows.extend([('M_SAME', *MASKED)] * (slots_per_bar - 1))
    return rows


def _constrain(logits, i, f0, banned, struct, instr, pad, voices):
    """generate.py's structural rules, vectorized over the whole canvas instead of one row."""
    # MASK is an input token and field 0's '' is padding: both are things the canvas is told,
    # never things it may answer, and nothing else stops a sampler from drawing one
    logits = logits.masked_fill(banned, float('-inf'))
    if i == 0:
        return logits
    is_struct = torch.isin(f0, struct).unsqueeze(-1)
    if i < 4:
        logits = logits.masked_fill(is_struct & ~pad, float('-inf'))       # structural rows carry '' in 1-3
        if i < 3:
            logits = logits.masked_fill(~is_struct & pad, float('-inf'))   # event rows always carry POS + a typed field 2
    else:
        is_instr = torch.isin(f0, instr).unsqueeze(-1)
        logits = logits.masked_fill(is_instr & ~voices, float('-inf'))     # an INSTR row names a voice
        logits = logits.masked_fill(is_struct & ~is_instr & ~pad, float('-inf'))
    return logits


def _onehot(size, ids, device):
    flags = torch.zeros(size, dtype=torch.bool, device=device)
    flags[list(ids)] = True
    return flags


@torch.no_grad()
def infill(model, tokenizer, rows, steps=12, temperature=1.0, gumbel=1.0, verbose=False):
    """Write every MASK in ``rows`` by confidence-ordered iterative decoding.

    Each pass samples the whole canvas at once, commits only the slots it is surest of, and
    re-masks the rest, so later passes condition on what earlier ones settled. Sampling every
    hidden slot independently ignores their joint distribution — committing a cosine-shrinking
    fraction per pass is what buys that dependence back. Observed fields are never resampled,
    so an injected row survives the run byte for byte.

    Returns (ids, order): the filled canvas, and each row's commit rank — 0 for the first slot
    the run settled, ascending, and -1 for a row the caller gave, which was never the model's
    to write. generation_labels turns those ranks into the labels the score carries."""
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor(tokenizer.encode(rows), dtype=torch.long, device=device).unsqueeze(0)

    mask_ids = torch.tensor(tokenizer.mask_ids, device=device)
    hidden = ids == mask_ids
    todo = hidden.any(-1)
    total = int(todo.sum())
    # the decode order is not the reading order and that is the whole point of recording it: a
    # slot's rank says when the model made up its mind, so a score labelled with it shows the
    # skeleton it committed early and the notes it only filled in once everything else was fixed
    order = torch.full(ids.shape[:2], -1, dtype=torch.long, device=device)
    settled = 0

    struct = torch.tensor(sorted(tokenizer.struct_f0), device=device)
    instr = torch.tensor(sorted(tokenizer.instr_f0), device=device)
    voices = _onehot(tokenizer.field_sizes[4], tokenizer.voice_f4, device)
    pads = [ _onehot(v, [tokenizer.pad_ids[i]], device) for i, v in enumerate(tokenizer.field_sizes) ]
    banned = [ _onehot(v, [tokenizer.mask_ids[i]] + ([tokenizer.pad_ids[i]] if i == 0 else []), device)
               for i, v in enumerate(tokenizer.field_sizes) ]

    for step in range(steps):
        # mask=None keeps SDPA on the dense flash kernel; is_causal=False is what lets a slot
        # read the injections after it as well as the ones before
        h = model.hidden(ids, is_causal=False)

        prev, logps = [], []
        for i in range(model.num_fields):
            logits = model.field_logits(i, h, prev).float()
            logits = _constrain(logits, i, prev[0] if i else None, banned[i],
                                struct, instr, pads[i], voices) / temperature
            pick = torch.multinomial(F.softmax(logits, -1).flatten(0, 1), 1).view(ids.shape[:2])
            # an observed field feeds the chain as itself, so heads above it condition on what
            # the caller actually injected rather than on a resampled guess
            pick = torch.where(hidden[..., i], pick, ids[..., i])
            logps.append(F.log_softmax(logits, -1).gather(-1, pick.unsqueeze(-1)).squeeze(-1))
            prev.append(pick)

        # only the hidden fields decide the ordering: an observed field's log-prob is a per-row
        # constant that would otherwise rank rows by how predictable their given fields are
        conf = (torch.stack(logps, -1) * hidden).sum(-1)
        # annealed Gumbel noise, so early passes explore and late ones commit on merit alone
        conf = conf - gumbel * (1 - step / steps) * torch.log(-torch.log(torch.rand_like(conf).clamp_min(1e-9)))
        conf = conf.masked_fill(~todo, float('-inf'))

        left = math.floor(total * math.cos(math.pi / 2 * (step + 1) / steps))
        take = int(todo.sum()) - left
        if take > 0:
            keep = torch.topk(conf, take, dim=-1).indices
            newly = torch.zeros_like(todo).scatter_(1, keep, True)
            ids = torch.where((newly.unsqueeze(-1) & hidden), torch.stack(prev, -1), ids)
            # topk hands the slots back in descending confidence, which IS the order this pass
            # resolved them in, so ranking inside a pass and across passes is one sequence
            # rather than a per-pass bucket the label would have to average over
            order[0, keep[0]] = torch.arange(settled, settled + take, device=device)
            settled += take
            todo &= ~newly
        if verbose:
            print(f'{step + 1}/{steps} | {int(todo.sum())} of {total} slots left')
        if not todo.any():
            break

    return ids[0].tolist(), order[0].tolist()


def repair_seams(model, tokenizer, canvas, filled, order, radius=2, **kwargs):
    """Rewrite the rows either side of every given row, now that the region around them exists.

    A slot committed on the first pass saw masks on one side of the join and a hard boundary on
    the other; this pass sees real music both ways. The audible artifacts sit at exactly those
    joins, so it is worth the second run.

    ``order`` is the first pass's ranks. A row this pass rewrites is renumbered after every
    rank the first pass handed out — what survives in it is what pass two wrote, so that is
    when it was decided — and a row left alone keeps the rank it already had."""
    given = { i for i, row in enumerate(canvas) if 'MASK' not in row }
    redo = { j for i in given for j in range(i - radius, i + radius + 1) } - given
    rows = [ (row[0], *MASKED) if i in redo and 0 <= i < len(filled) else row
             for i, row in enumerate(tokenizer.decode(filled)) ]
    repaired, reorder = infill(model, tokenizer, rows, **kwargs)
    base = max(order) + 1
    return repaired, [ base + new if new >= 0 else old for new, old in zip(reorder, order) ]


def generation_labels(order):
    """Commit ranks renumbered densely from 1 over the rows that survive the slack filter.

    A rank counts every slot the run settled, NULL slack included, so on a canvas that is
    mostly slack the raw numbers come out as a sparse scatter of four-digit labels. Renumbering
    the survivors preserves the relative order exactly while making the score readable as
    "1st decided, 2nd decided, ...". A caller-given row stays None: it was never generated, and
    leaving it unlabelled is the clearest way for the score to say so."""
    rank = { o: i for i, o in enumerate(sorted(o for o in order if o >= 0), start=1) }
    return [ rank.get(o) for o in order ]


def canvas_to_tokens(tokenizer, ids, order=None):
    """Decoded rows with the slack dropped, alongside their generation labels. resolve_bars runs
    FIRST: it walks a running bar counter, so a NULL row that landed on an M_NEXT slot still has
    to advance it — strip the NULLs before resolving and every bar after that point silently
    shifts down by one."""
    rows = resolve_bars(tokenizer.decode(ids))
    kept = [ i for i, row in enumerate(rows) if 'NULL' not in row ]
    # renumbered after the filter, not before: the slack is most of the canvas and most of the
    # ranks with it, so labelling first would leave the survivors scattered across that range
    labels = generation_labels([ order[i] for i in kept ]) if order is not None else None
    return [ rows[i] for i in kept ], labels


if __name__ == "__main__":
    tokenizer = QuintupleTokenizer()
    model, _ = load_model_only('the-waffiller')

    # a header, eight bars of nothing, and a cadence to land on: the model writes the middle
    canvas = [
        ('SOG', '', '', '', ''),
        ('INSTR_piano', '', '', '', 'VOICE_0'),
        ('INSTR_piano', '', '', '', 'VOICE_1'),
        ('SOS', '', '', '', ''),
        ('M_NEXT', 'POS_0', 'KEY_0', '', ''),
        ('M_SAME', 'POS_0', 'TEMPO_8', '', ''),
        ('M_SAME', 'POS_0', 'NUM_4', 'DEN_4', ''),
        ('M_SAME', 'POS_0', 'CLEF_treble', '', 'VOICE_0'),
        ('M_SAME', 'POS_0', 'CLEF_bass', '', 'VOICE_1'),
        ('M_SAME', 'POS_0', 'DEG_11', 'ROOT_11', 'QUAL_maj7'),
        ('M_SAME', 'POS_0', 'DUR_48', 'MASK', 'VOICE_0'),
        ('M_SAME', 'POS_0', 'DUR_48', 'MASK', 'VOICE_0'),
        ('M_SAME', 'POS_0', 'DUR_48', 'MASK', 'VOICE_0'),
        ('M_SAME', 'POS_0', 'DUR_48', 'MASK', 'VOICE_1'),
        # *masked_bars(bars=1, slots_per_bar=16, start_next=False),
        ('M_SAME', 'POS_48', 'DEG_2', 'ROOT_2', 'QUAL_7'),
        ('M_SAME', 'POS_48', 'DUR_48', 'MASK', 'VOICE_0'),
        ('M_SAME', 'POS_48', 'DUR_48', 'MASK', 'VOICE_0'),
        ('M_SAME', 'POS_48', 'DUR_48', 'MASK', 'VOICE_0'),
        ('M_SAME', 'POS_48', 'DUR_48', 'MASK', 'VOICE_1'),
        # *masked_bars(bars=1, slots_per_bar=16, start_next=False),
        ('M_NEXT', 'POS_0', 'DEG_7', 'ROOT_7', 'QUAL_maj7'),
        ('M_SAME', 'POS_0', 'DUR_48', 'MASK', 'VOICE_0'),
        ('M_SAME', 'POS_0', 'DUR_48', 'MASK', 'VOICE_0'),
        ('M_SAME', 'POS_0', 'DUR_48', 'MASK', 'VOICE_0'),
        ('M_SAME', 'POS_0', 'DUR_48', 'MASK', 'VOICE_1'),
        # *masked_bars(bars=1, slots_per_bar=16, start_next=False),
        ('M_SAME', 'POS_48', 'DEG_10', 'ROOT_10', 'QUAL_7'),
        ('M_SAME', 'POS_48', 'DUR_48', 'MASK', 'VOICE_0'),
        ('M_SAME', 'POS_48', 'DUR_48', 'MASK', 'VOICE_0'),
        ('M_SAME', 'POS_48', 'DUR_48', 'MASK', 'VOICE_0'),
        ('M_SAME', 'POS_48', 'DUR_48', 'MASK', 'VOICE_1'),
        # *masked_bars(bars=1, slots_per_bar=16, start_next=False),
        ('M_NEXT', 'POS_0', 'DEG_3', 'ROOT_3', 'QUAL_maj7'),
        ('M_SAME', 'POS_0', 'DUR_96', 'MASK', 'VOICE_0'),
        ('M_SAME', 'POS_0', 'DUR_96', 'MASK', 'VOICE_0'),
        ('M_SAME', 'POS_0', 'DUR_96', 'MASK', 'VOICE_0'),
        ('M_SAME', 'POS_0', 'DUR_96', 'MASK', 'VOICE_1'),
        # *masked_bars(bars=1, slots_per_bar=16, start_next=False),
        ('EOS', '', '', '', ''),
    ]
    print(canvas)
    filled, order = infill(model, tokenizer, canvas, verbose=True, temperature=0.5)
    repaired, order = repair_seams(model, tokenizer, canvas, filled, order)
    tokens, labels = canvas_to_tokens(tokenizer, repaired, order)
    # printed in reading order with the generation order beside it, so the console says the same
    # thing the score does; '-' is a row the canvas gave rather than one the model wrote
    print('\n'.join(f'{"-" if label is None else label:>4}  {token}'
                    for label, token in zip(labels, tokens)))
    tokens_to_score(tokens, labels)
    print('Exported')
