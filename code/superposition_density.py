"""
How many superposed token-shaped signals does the residual stream carry?

Measurement 1: ITERATIVE PEEL ON RAW h
  Take h (not Δh) at a layer. Project through W_U, read top-k tokens.
  Project h onto those W_U rows, subtract, re-read the remainder.
  Count how many rounds produce coherent (human-readable) tokens
  before the readout becomes garbage.

Measurement 2: ACTIVE MLP NEURON COUNT
  How many neurons are ON (post-GELU > threshold) at each layer?
  This is an upper bound on features being processed.

Measurement 3: W_U LOGIT DISTRIBUTION
  How peaked or spread is h @ W_U.T? The effective number of tokens
  that h "points at" — measured by softmax entropy.

Test across several prompts and layers.
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")

print(f"Loading {MODEL} on {DEV}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)

NL = model.config.num_hidden_layers
d_model = model.config.hidden_size
d_inter = model.config.intermediate_size
W_U = model.lm_head.weight.detach().float()


def _sl(*layers):
    return sorted(set(min(round(l * NL / 32), NL) for l in layers))


def topk_tok(logits, k=5):
    vals, idxs = torch.topk(logits.float(), k)
    return [(tok.decode([int(idxs[j])]).strip()[:14], f"{float(vals[j]):.1f}")
            for j in range(k)]


def get_hidden_and_mlp(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    mlp_acts = {}

    hooks = []
    for L in range(NL):
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                mlp_acts[layer_idx] = output[0, -1, :].detach().float()
            return hook_fn
        h = model.model.layers[L].mlp.fc1.register_forward_hook(make_hook(L))
        hooks.append(h)

    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV), output_hidden_states=True)

    for h in hooks:
        h.remove()

    # Apply GELU to get post-activation
    post_gelu = {}
    for L in mlp_acts:
        gelu = model.model.layers[L].mlp.activation_fn
        post_gelu[L] = gelu(mlp_acts[L])

    return out, post_gelu


prompts = [
    "The hot dog was",
    "She caught a cold and went to",
    "The capital of France is",
    "When Mary and John went to the store, John gave a drink to",
    "The ice in the bucket was extremely cold. The temperature was",
    "Some of the students passed the exam, so",
    "The movie was absolutely wonderful and everyone",
    "He slipped a bottle under his coat and walked out without paying. He",
    "def calculate_sum(a, b):\n    return a +",
    "The elephant walked slowly, its massive body",
]

layers = _sl(4, 8, 12, 16, 20, 24, 28)


# ══════════════════════════════════════════════════════════════
# MEASUREMENT 1: ITERATIVE PEEL ON RAW h
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("MEASUREMENT 1: ITERATIVE PEEL — how many token layers in raw h?")
print("=" * 70)

K_PER_ROUND = 10  # tokens to peel per round
N_ROUNDS = 10

for prompt in prompts[:6]:
    print(f"\n  \"{prompt[-50:]}\"")
    out, _ = get_hidden_and_mlp(prompt)

    for L in [_sl(12)[0], _sl(20)[0], _sl(28)[0]]:
        h = out.hidden_states[L][0, -1, :].float().to(DEV)
        remainder = h.clone()

        print(f"    L{L}:")
        for r in range(N_ROUNDS):
            logits = remainder @ W_U.T
            top = topk_tok(logits, 5)
            norm = float(remainder.norm())

            # Get top-K token directions
            topk_idx = torch.topk(logits.abs(), K_PER_ROUND).indices
            wu_rows = W_U[topk_idx]
            Q, R = torch.linalg.qr(wu_rows.T)
            proj = Q @ (Q.T @ remainder)
            remainder = remainder - proj

            top_str = ', '.join(t[0] for t in top[:4])
            print(f"      round {r}: [{top_str}]  ||={norm:.0f}")

    del out
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# MEASUREMENT 2: ACTIVE MLP NEURON COUNT
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MEASUREMENT 2: ACTIVE MLP NEURONS per layer")
print("=" * 70)

thresholds = [0.01, 0.1, 0.3, 1.0]

for prompt in prompts[:6]:
    print(f"\n  \"{prompt[-50:]}\"")
    out, post_gelu = get_hidden_and_mlp(prompt)

    for L in layers:
        if L not in post_gelu:
            continue
        acts = post_gelu[L]
        counts = []
        for thresh in thresholds:
            n_active = int((acts.abs() > thresh).sum())
            counts.append(n_active)

        print(f"    L{L:>2}: " + "  ".join(
            f">{t}:{n:>5} ({n/d_inter*100:.0f}%)"
            for t, n in zip(thresholds, counts)))

    del out, post_gelu
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# MEASUREMENT 3: W_U LOGIT DISTRIBUTION — entropy of raw h
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MEASUREMENT 3: EFFECTIVE TOKEN COUNT via softmax entropy")
print("=" * 70)
print("Entropy of softmax(h @ W_U.T) → effective number of tokens = exp(H)")

for prompt in prompts[:6]:
    print(f"\n  \"{prompt[-50:]}\"")
    out, _ = get_hidden_and_mlp(prompt)

    for L in layers:
        h = out.hidden_states[L][0, -1, :].float().to(DEV)
        logits = h @ W_U.T
        probs = torch.softmax(logits, dim=-1)
        entropy = -float((probs * torch.log(probs + 1e-10)).sum())
        eff_tokens = math.exp(entropy)

        # Also measure for logit lens (with final LN)
        top = topk_tok(logits, 4)
        top_str = ', '.join(t[0] for t in top)

        print(f"    L{L:>2}: H={entropy:.1f} nats  eff_tokens={eff_tokens:.0f}  "
              f"top=[{top_str}]")

    del out
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# MEASUREMENT 4: PEEL COHERENCE — when does peeling hit garbage?
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MEASUREMENT 4: PEEL COHERENCE — max logit magnitude per round")
print("=" * 70)
print("Tracks peak logit per round. Sharp drop = out of readable tokens.\n")

prompt = "The hot dog was"
out, _ = get_hidden_and_mlp(prompt)

for L in layers:
    h = out.hidden_states[L][0, -1, :].float().to(DEV)
    remainder = h.clone()

    peaks = []
    for r in range(15):
        logits = remainder @ W_U.T
        peak = float(logits.abs().max())
        peaks.append(peak)

        topk_idx = torch.topk(logits.abs(), K_PER_ROUND).indices
        wu_rows = W_U[topk_idx]
        Q, R = torch.linalg.qr(wu_rows.T)
        proj = Q @ (Q.T @ remainder)
        remainder = remainder - proj

    # Normalize to round 0
    if peaks[0] > 0:
        norm_peaks = [p / peaks[0] for p in peaks]
    else:
        norm_peaks = peaks

    bar_str = "  ".join(f"{p:.2f}" for p in norm_peaks[:10])
    print(f"  L{L:>2}: {bar_str}")

del out
torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
