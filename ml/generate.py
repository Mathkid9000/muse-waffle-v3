import torch
import torch.nn.functional as F
from helpers import SEQ_LEN
from model_util import create_causal_mask, load_model_only
from tokenizer import QuintupleTokenizer, tokens_to_score


def sample_field(logits, temperature, top_k):
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k:
        k = min(top_k, logits.size(-1))
        kept, idx = torch.topk(logits, k)
        probs = F.softmax(kept, dim=-1)
        return int(idx[torch.multinomial(probs, 1)])
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1))


@torch.no_grad()
def generate_score(model, tokenizer, seed, seq_len=SEQ_LEN, temperature=1.0, top_k=30):
    model.eval()
    eos_id = tokenizer.vocabs[0]['EOS']

    tokens = torch.tensor(seed, dtype=torch.long, device=device)

    while tokens.shape[0] < seq_len:
        print(f"{tokens.shape[0]} of {seq_len} | {tokenizer.inv[0].get(int(tokens[-1][0]))}")
        ctx = tokens[-seq_len:].unsqueeze(0)
        mask = create_causal_mask(ctx.shape[1], device).unsqueeze(0)
        h_last = model.hidden(ctx, mask)[:, -1]

        # Intra-row chain: field 0 first, later fields conditioned on the
        # row's earlier fields (same factorization RL sampled/trained under).
        prev = []
        for i in range(model.num_fields):
            field_logits = model.field_logits(i, h_last, prev)
            fid = sample_field(field_logits[0], temperature, top_k)
            prev.append(torch.tensor([fid], dtype=torch.long, device=device))
        next_tok = torch.cat(prev)
        tokens = torch.cat([tokens, next_tok.unsqueeze(0)], dim=0)

        if next_tok[0].item() == eos_id:
            break

    return tokens.cpu().numpy()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seed
    seed = [
        ('SOG', '', '', '', '')
    ]
    tokenizer = QuintupleTokenizer()
    seed_encoded = tokenizer.encode(seed)

    # model
    model, _ = load_model_only('TEST123', device)

    # prediction
    encoded_ids = generate_score(model, tokenizer, seed_encoded, temperature=0.8, top_k=30)
    decoded_ids = tokenizer.decode(encoded_ids)
    print(f'Generated {len(decoded_ids)} of {SEQ_LEN}')
    print('\n'.join([ str(t) for t in decoded_ids ]))
    tokens_to_score(decoded_ids)
    print('Exported')

