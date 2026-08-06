import math
import torch
import torch.nn.functional as F

from model_util import load_model_only
from tokenizer import QuintupleTokenizer, resolve_bars, tokens_to_score

MASKED = ('MASK', 'MASK', 'MASK', 'MASK')


def masked_bars(bars, slots_per_bar):
    """The canvas a caller hands over for a region it wants written: one bar advance per bar,
    the rest of that bar's slots left at the same bar, every other field the model's to choose.

    Over-allocate ``slots_per_bar`` — the model empties the slack into NULL rows, but it cannot
    add slots it was never given, so a tight budget caps the region's event density."""
    rows = []
    for _ in range(bars):
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
    so an injected row survives the run byte for byte."""
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor(tokenizer.encode(rows), dtype=torch.long, device=device).unsqueeze(0)

    mask_ids = torch.tensor(tokenizer.mask_ids, device=device)
    hidden = ids == mask_ids
    todo = hidden.any(-1)
    total = int(todo.sum())

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
            todo &= ~newly
        if verbose:
            print(f'{step + 1}/{steps} | {int(todo.sum())} of {total} slots left')
        if not todo.any():
            break

    return ids[0].tolist()


def repair_seams(model, tokenizer, canvas, filled, radius=2, **kwargs):
    """Rewrite the rows either side of every given row, now that the region around them exists.

    A slot committed on the first pass saw masks on one side of the join and a hard boundary on
    the other; this pass sees real music both ways. The audible artifacts sit at exactly those
    joins, so it is worth the second run."""
    given = { i for i, row in enumerate(canvas) if 'MASK' not in row }
    redo = { j for i in given for j in range(i - radius, i + radius + 1) } - given
    rows = [ (row[0], *MASKED) if i in redo and 0 <= i < len(filled) else row
             for i, row in enumerate(tokenizer.decode(filled)) ]
    return infill(model, tokenizer, rows, **kwargs)


def canvas_to_tokens(tokenizer, ids):
    """Decoded rows with the slack dropped. resolve_bars runs FIRST: it walks a running bar
    counter, so a NULL row that landed on an M_NEXT slot still has to advance it — strip the
    NULLs before resolving and every bar after that point silently shifts down by one."""
    rows = resolve_bars(tokenizer.decode(ids))
    return [ row for row in rows if 'NULL' not in row ]


if __name__ == "__main__":
    tokenizer = QuintupleTokenizer()
    model, _ = load_model_only('TEST123')

    # a header, eight bars of nothing, and a cadence to land on: the model writes the middle
    canvas = [
        ('SOG', '', '', '', ''),
        ('INSTR_piano', '', '', '', 'VOICE_0'),
        ('INSTR_piano', '', '', '', 'VOICE_1'),
        ('SOS', '', '', '', ''),
        *masked_bars(bars=8, slots_per_bar=12),
        ('M_NEXT', 'POS_0', 'DUR_96', 'PITCH_60', 'VOICE_0'),
        ('EOS', '', '', '', ''),
    ]
    filled = infill(model, tokenizer, canvas, verbose=True)
    tokens = canvas_to_tokens(tokenizer, repair_seams(model, tokenizer, canvas, filled))
    print('\n'.join(str(t) for t in tokens))
    tokens_to_score(tokens)
    print('Exported')
