import torch
import torch.nn.functional as F
from helpers import SEQ_LEN
from model_util import load_model_only
from tokenizer import QuintupleTokenizer, tokens_to_score


def sample_field(logits, temperature, top_k, top_p=0.9):
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k:
        k = min(top_k, logits.size(-1))
        kept, idx = torch.topk(logits, k)
    else:
        kept, idx = logits, torch.arange(logits.size(-1), device=logits.device)
    probs = F.softmax(kept, dim=-1)
    if top_p and top_p < 1.0:
        # nucleus: keep the smallest prefix of the sorted distribution holding top_p mass, so the
        # tail top-k lets through on flat distributions (where k reaches far down) is cut, while
        # confident distributions keep their handful of real candidates
        sorted_probs, order = torch.sort(probs, descending=True)
        cut = int((torch.cumsum(sorted_probs, dim=-1) < top_p).sum().item()) + 1
        nucleus = torch.zeros_like(probs)
        nucleus[order[:cut]] = probs[order[:cut]]
        probs = nucleus / nucleus.sum()
    return int(idx[torch.multinomial(probs, 1)])


@torch.no_grad()
def generate_score(model, tokenizer, seed, max_events=SEQ_LEN, seq_len=SEQ_LEN,
                   temperature=1.0, top_k=30, top_p=0.9, min_events=200, verbose=False):
    """Sample rows autoregressively until EOS or ``max_events``. The context is the last
    ``seq_len`` rows, so ``max_events`` may exceed the model's window: with relative bar
    tokens a sliding window stays coherent, since no row references an absolute position.

    Each row is drawn through the intra-row chain (field 0 first, later fields conditioned
    on the row's earlier fields — the same factorization training scored), with structural
    constraints the corpus obeys by construction masked in rather than left to the sampler:
    structural rows (SOG/SOS/EOS/INSTR) carry '' in fields 1-3 and '' or a VOICE in field 4,
    event rows always carry a real POS and typed field 2, and EOS can't close the piece
    before ``min_events`` rows.
    """
    model.eval()
    device = next(model.parameters()).device
    eos_id = tokenizer.vocabs[0]['EOS']
    empty_ids = tokenizer.pad_ids           # id of '' per field (index 0 unused)
    voice_ids = torch.tensor(sorted(tokenizer.voice_f4), dtype=torch.long, device=device)

    tokens = torch.tensor(seed, dtype=torch.long, device=device)

    while tokens.shape[0] < max_events:
        if verbose:
            print(f"{tokens.shape[0]} of {max_events} | {tokenizer.inv[0].get(int(tokens[-1][0]))}")
        # mask=None routes GQA to is_causal flash SDPA; an explicit bool mask would fall off
        # the flash kernel and materialize the full TxT score matrix on every generated row
        ctx = tokens[-seq_len:].unsqueeze(0)
        h_last = model.hidden(ctx)[:, -1]

        prev = []
        f0_is_struct = f0_is_instr = False
        for i in range(model.num_fields):
            # a structural row's fields 1-3 are '' by construction; force them rather than
            # trusting the sampled tail, which top-k otherwise pollutes with note-like garbage
            if i > 0 and f0_is_struct and not (i == 4 and f0_is_instr):
                fid = empty_ids[i]
                prev.append(torch.tensor([fid], dtype=torch.long, device=device))
                continue

            field_logits = model.field_logits(i, h_last, prev)[0]
            if i == 0 and tokens.shape[0] < min_events:
                field_logits[eos_id] = float('-inf')    # too short to close the piece
            if i in (1, 2) and not f0_is_struct:
                field_logits[empty_ids[i]] = float('-inf')  # event rows always carry POS + typed f2
            if i == 4 and f0_is_instr:
                masked = torch.full_like(field_logits, float('-inf'))
                masked[voice_ids] = field_logits[voice_ids]  # an INSTR row names a voice
                field_logits = masked

            fid = sample_field(field_logits, temperature, top_k, top_p)
            if i == 0:
                f0_is_struct = fid in tokenizer.struct_f0
                f0_is_instr = fid in tokenizer.instr_f0
            prev.append(torch.tensor([fid], dtype=torch.long, device=device))

        next_tok = torch.cat(prev)
        tokens = torch.cat([tokens, next_tok.unsqueeze(0)], dim=0)

        if next_tok[0].item() == eos_id:
            break

    return tokens.cpu().numpy()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seed with a header (instrumentation + start-of-score) instead of a bare SOG: sampling the
    # header at temperature wanders through instrument combinations, and a generation that never
    # emits an INSTR row can't be decoded into a score at all
    seed = [
        ('SOG', '', '', '', ''),
        ('INSTR_piano', '', '', '', 'VOICE_0'),
        ('SOS', '', '', '', ''),
    ]
    tokenizer = QuintupleTokenizer()
    seed_encoded = tokenizer.encode(seed)

    # model
    model, _ = load_model_only('TEST123', device)

    # prediction
    encoded_ids = generate_score(model, tokenizer, seed_encoded, temperature=0.8, top_k=30, top_p=0.9, verbose=True)
    decoded_ids = tokenizer.decode(encoded_ids)
    print(f'Generated {len(decoded_ids)} of {SEQ_LEN}')
    print('\n'.join([ str(t) for t in decoded_ids ]))
    tokens_to_score(decoded_ids)
    print('Exported')
