import torch
from torch import nn
import torch.nn.functional as F


class GQA(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(hidden_dim, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, hidden_dim)

    def forward(self, q, k, v, mask=None, rope=None):
        q_batch_size, q_seq_len, hidden_dim = q.shape
        k_batch_size, k_seq_len, hidden_dim = k.shape
        v_batch_size, v_seq_len, hidden_dim = v.shape

        # projection
        q = self.q_proj(q).view(q_batch_size, q_seq_len, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(k_batch_size, k_seq_len, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(v_batch_size, v_seq_len, -1, self.head_dim).transpose(1, 2)

        # apply rotary positional encoding
        if rope:
            q = rope(q)
            k = rope(k)

        # Expand the KV heads to match the query heads so grouped-query
        # attention works without SDPA's enable_gqa flag (older torch).
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # compute grouped query attention
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        # a mask handed over as a tensor rules out SDPA's flash kernel, which is the one that
        # never materializes the T x T scores; asking for causal by name instead keeps attention
        # memory linear in the sequence length, so pass no mask unless it isn't just causal
        output = F.scaled_dot_product_attention(q, k, v,
                                                attn_mask=mask,
                                                is_causal=mask is None,
                                                dropout_p=self.dropout if self.training else 0.0)
        output = output.transpose(1, 2).reshape(q_batch_size, q_seq_len, hidden_dim).contiguous()
        output = self.out_proj(output)
        return output


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, intermediate_dim)
        self.up = nn.Linear(hidden_dim, intermediate_dim)
        self.down = nn.Linear(intermediate_dim, hidden_dim)
        self.act = nn.SiLU()
 
    def forward(self, x):
        x = self.act(self.gate(x)) * self.up(x)
        x = self.down(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * x).to(input_dtype)
nn.RMSNorm = RMSNorm


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_len).float()
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())

    def forward(self, x, seq_len=None):
        # x is (batch, num_heads, seq_len, head_dim) at the call site in GQA,
        # so the sequence axis is dim -2 and cos/sin broadcast over the heads.
        if seq_len is None:
            seq_len = x.size(-2)
        cos = self.cos[:seq_len].view(1, 1, seq_len, -1)
        sin = self.sin[:seq_len].view(1, 1, seq_len, -1)
        return apply_rotary_pos_emb(x, cos, sin)


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads, dropout=0.1,
                 intermediate_dim=None):
        super().__init__()
        # SwiGLU spends three matrices where a plain GELU MLP spends two, so the
        # usual 4x width is really 1.5x the parameters that convention sizes for;
        # 8/3x is the width that holds them equal. Defaulting to 4x here keeps
        # configs written before intermediate_dim existed loading unchanged.
        if intermediate_dim is None:
            intermediate_dim = 4 * hidden_dim
        self.self_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.mlp = SwiGLU(hidden_dim, intermediate_dim)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None, rope=None):
        # self-attention sublayer
        out = self.norm1(x)
        out = self.self_attn(out, out, out, mask, rope)
        x = self.drop(out) + x
        # MLP sublayer
        out = self.norm2(x)
        out = self.mlp(out)
        return self.drop(out) + x


class MusicTransformer(nn.Module):
    """Compound / factorized decoder for quintuple music tokens.

    Each of the ``len(field_sizes)`` fields is embedded separately; the field
    embeddings are concatenated and projected to ``hidden_dim`` before the
    transformer stack.

    The output heads form an intra-row autoregressive chain (CP-word style):
    head i predicts field i from the hidden state concatenated with the
    embeddings of fields 0..i-1 of the SAME row being predicted. Independent
    heads cannot represent within-row consistency whenever the context is ambiguous about
    the row's type — sampling then produces field combinations that never occur
    in the data. The chain makes p(field_i | context, field_<i) learnable, so
    the joint row distribution factorizes exactly.
    """

    def __init__(self, num_layers, num_heads, num_kv_heads, hidden_dim,
                 max_seq_len, field_sizes, embed_dims, dropout=0.1, epoches=[],
                 intermediate_dim=None):
        super().__init__()
        assert len(field_sizes) == len(embed_dims)
        self.num_fields = len(field_sizes)
        self.rope = RotaryPositionalEncoding(hidden_dim // num_heads, max_seq_len)
        self.field_embeddings = nn.ModuleList([
            nn.Embedding(vocab, dim) for vocab, dim in zip(field_sizes, embed_dims)
        ])
        self.input_proj = nn.Linear(sum(embed_dims), hidden_dim)
        self.embed_drop = nn.Dropout(dropout)
        self.decoders = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, num_kv_heads, dropout, intermediate_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_dim)
        # Head i sees [hidden ; emb(field_0) ; ... ; emb(field_{i-1})]; the
        # conditioning reuses the input embedding tables (shared weights).
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim + sum(embed_dims[:i]), vocab)
            for i, vocab in enumerate(field_sizes)
        ])

    def hidden(self, ids, mask=None):
        """Trunk only: (B, T, num_fields) ids -> (B, T, hidden_dim)."""
        embeds = [emb(ids[..., i]) for i, emb in enumerate(self.field_embeddings)]
        x = self.embed_drop(self.input_proj(torch.cat(embeds, dim=-1)))
        for decoder in self.decoders:
            x = decoder(x, mask, self.rope)
        return self.norm(x)

    def field_logits(self, i, h, prev_fields):
        """Logits for field ``i`` given hidden state ``h`` (..., hidden_dim) and
        the row's already-chosen earlier fields ``prev_fields`` (list of i long
        tensors broadcastable to h's leading dims). Used at sampling time."""
        cond = [h] + [self.field_embeddings[j](prev_fields[j]) for j in range(i)]
        return self.heads[i](torch.cat(cond, dim=-1))

    def forward(self, ids, mask=None, target_fields=None):
        """Teacher-forced logits: one (B, T, field_size_i) tensor per field.

        ``target_fields`` (B, T, num_fields) is the row each position predicts
        (the usual one-step-shifted targets). Head i conditions on that row's
        fields < i — the within-row chain rule; no cross-row leak because
        position t is being scored on exactly that row anyway.
        """
        if target_fields is None:
            raise ValueError(
                "forward() needs target_fields for the intra-row chain; "
                "use hidden() + field_logits() for step-wise sampling")
        h = self.hidden(ids, mask)
        return [self.field_logits(i, h, [target_fields[..., j] for j in range(i)])
                for i in range(self.num_fields)]