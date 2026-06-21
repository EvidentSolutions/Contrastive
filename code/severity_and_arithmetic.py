"""
Two experiments:
1. Killing severity: fly/spider/chicken/dog/boy/woman — how does the
   model's continuation differ, and what does prefix-contrasting reveal?
2. Simple arithmetic: contrastive pairs like 2+3 vs 2+5

Usage: .venv/Scripts/python.exe contrastive/code/severity_and_arithmetic.py
"""
import sys
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")


def tk(logits, k=6):
    v, i = torch.topk(logits, k)
    return ", ".join(
        tok.decode([int(i[j])]).strip()[:12] for j in range(k)
    )


def bk(logits, k=6):
    v, i = torch.topk(logits, k, largest=False)
    return ", ".join(
        tok.decode([int(i[j])]).strip()[:12] for j in range(k)
    )


print(f"Loading {MODEL}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
for p in model.parameters():
    p.requires_grad_(False)

NL = model.config.num_hidden_layers
W_U = model.lm_head.weight.detach()


def _sl(*layers):
    """Scale layer indices from 32-layer base to current NL."""
    return sorted(set(min(round(l * NL / 32), NL) for l in layers))

# ============================================================
# 1. KILLING SEVERITY
# ============================================================
print("=" * 100)
print("1. KILLING SEVERITY")
print("=" * 100)

prefix = "The man killed a"
objects = ["fly", "spider", "chicken", "dog", "boy", "woman", "stranger"]

# Get prefix hidden state
prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
with torch.no_grad():
    prefix_out = model(
        torch.tensor([prefix_ids], device=DEV),
        output_hidden_states=True,
    )

for obj in objects:
    text = f"{prefix} {obj}"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(
            torch.tensor([ids], device=DEV),
            output_hidden_states=True,
        )
        gen = model.generate(
            torch.tensor([ids], device=DEV),
            max_new_tokens=25,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    answer = tok.decode(gen[0][len(ids) :]).strip().split("\n")[0][:60]
    print(f'\n  "{text}" -> "{answer}"')

    # Contrast at the OBJECT position vs the prefix's last position
    # prefix ends at 'a' (pos 3), full prompt has object at pos 4
    for L in _sl(20, 28):
        h_obj = out.hidden_states[L][0, -1, :].float()
        h_pre = prefix_out.hidden_states[L][0, -1, :].float()
        dh = h_obj - h_pre
        norm = float(dh.norm() / h_obj.norm())
        ld = dh @ W_U.float().T
        print(
            f"    L{L} last_pos contrast: ({norm:.3f}) [{tk(ld)}]"
        )

    del out

# Direct pairwise: fly vs boy
print("\n--- FLY vs BOY direct ---")
fly_ids = tok("The man killed a fly", add_special_tokens=False)[
    "input_ids"
]
boy_ids = tok("The man killed a boy", add_special_tokens=False)[
    "input_ids"
]

with torch.no_grad():
    fly_out = model(
        torch.tensor([fly_ids], device=DEV),
        output_hidden_states=True,
    )
    boy_out = model(
        torch.tensor([boy_ids], device=DEV),
        output_hidden_states=True,
    )

for L in _sl(8, 16, 20, 24, 28, 32):
    # At object position (pos 5)
    h_f = fly_out.hidden_states[L][0, -1, :].float()
    h_b = boy_out.hidden_states[L][0, -1, :].float()
    dh = h_f - h_b
    norm = float(dh.norm() / h_f.norm())
    ld = dh @ W_U.float().T
    print(f"  L{L:>2} at obj pos: ({norm:.3f}) fly=[{tk(ld)}]  boy=[{bk(ld)}]")

del fly_out, boy_out, prefix_out
torch.cuda.empty_cache()

# ============================================================
# 2. ARITHMETIC
# ============================================================
print(f"\n{'='*100}")
print("2. ARITHMETIC")
print("=" * 100)

# Correctness check
arith = [
    "2 + 3 =",
    "2 + 5 =",
    "7 + 8 =",
    "12 + 13 =",
    "99 + 1 =",
    "5 - 3 =",
    "10 - 7 =",
    "3 * 4 =",
    "6 * 7 =",
]

print("\nCorrectness:")
for expr in arith:
    ids = tok(expr, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        gen = model.generate(
            torch.tensor([ids], device=DEV),
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    answer = tok.decode(gen[0][len(ids) :]).strip().split("\n")[0][:20]
    print(f"  {expr:>12} -> {answer}")

# Contrastive pairs
arith_pairs = [
    ("2+3 vs 2+5", "2 + 3 =", "2 + 5 ="),
    ("2+3 vs 3+3", "2 + 3 =", "3 + 3 ="),
    ("2+3 vs 2-3", "2 + 3 =", "2 - 3 ="),
    ("7+8 vs 7+2", "7 + 8 =", "7 + 2 ="),
    ("3*4 vs 3*5", "3 * 4 =", "3 * 5 ="),
]

for label, pa, pb in arith_pairs:
    ids_a = tok(pa, add_special_tokens=False)["input_ids"]
    ids_b = tok(pb, add_special_tokens=False)["input_ids"]

    with torch.no_grad():
        out_a = model(
            torch.tensor([ids_a], device=DEV),
            output_hidden_states=True,
        )
        out_b = model(
            torch.tensor([ids_b], device=DEV),
            output_hidden_states=True,
        )

    # Predictions
    probs_a = torch.softmax(out_a.logits[0, -1].float(), -1)
    probs_b = torch.softmax(out_b.logits[0, -1].float(), -1)
    va, ia = torch.topk(probs_a, 3)
    vb, ib = torch.topk(probs_b, 3)
    top_a = [
        (tok.decode([int(ia[j])]).strip(), round(float(va[j]), 3))
        for j in range(3)
    ]
    top_b = [
        (tok.decode([int(ib[j])]).strip(), round(float(vb[j]), 3))
        for j in range(3)
    ]

    print(f'\n  {label}:')
    print(f'    A: "{pa}" -> {top_a}')
    print(f'    B: "{pb}" -> {top_b}')
    print(
        f"    Tokens A: "
        f"{[tok.decode([t]).strip() for t in ids_a]}"
    )
    print(
        f"    Tokens B: "
        f"{[tok.decode([t]).strip() for t in ids_b]}"
    )

    for L in _sl(16, 24, 28, 32):
        h_a = out_a.hidden_states[L][0, -1, :].float()
        h_b = out_b.hidden_states[L][0, -1, :].float()
        dh = h_a - h_b
        norm = float(dh.norm() / h_a.norm())
        ld = dh @ W_U.float().T
        print(f"    L{L:>2} ({norm:.3f}) A=[{tk(ld)}]  B=[{bk(ld)}]")

    del out_a, out_b

# Track specific answer tokens across layers for 2+3=5 vs 2+5=7
print("\n--- Answer token tracking: 2+3 vs 2+5 ---")
ids_23 = tok("2 + 3 =", add_special_tokens=False)["input_ids"]
ids_25 = tok("2 + 5 =", add_special_tokens=False)["input_ids"]

tok5 = tok(" 5", add_special_tokens=False)["input_ids"][0]
tok7 = tok(" 7", add_special_tokens=False)["input_ids"][0]

with torch.no_grad():
    out_23 = model(
        torch.tensor([ids_23], device=DEV),
        output_hidden_states=True,
    )
    out_25 = model(
        torch.tensor([ids_25], device=DEV),
        output_hidden_states=True,
    )

print(f"  Tracking ' 5' (id={tok5}) and ' 7' (id={tok7})")
print(f"  {'L':>3} | 2+3: P(5)    P(7)  rank5 | 2+5: P(5)    P(7)  rank7")

for L in range(0, NL + 1, 4):
    h23 = out_23.hidden_states[L][0, -1, :].float()
    h25 = out_25.hidden_states[L][0, -1, :].float()
    l23 = h23 @ W_U.float().T
    l25 = h25 @ W_U.float().T

    p23_5 = float(torch.softmax(l23, -1)[tok5])
    p23_7 = float(torch.softmax(l23, -1)[tok7])
    r23_5 = int((l23 > l23[tok5]).sum().item()) + 1

    p25_5 = float(torch.softmax(l25, -1)[tok5])
    p25_7 = float(torch.softmax(l25, -1)[tok7])
    r25_7 = int((l25 > l25[tok7]).sum().item()) + 1

    print(
        f"  {L:>3} | "
        f"     {p23_5:.4f}  {p23_7:.4f}  r{r23_5:<5} | "
        f"     {p25_5:.4f}  {p25_7:.4f}  r{r25_7}"
    )

del out_23, out_25
torch.cuda.empty_cache()

print(f"\n{'='*100}")
print("DONE")
