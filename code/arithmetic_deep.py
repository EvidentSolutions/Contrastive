"""
Systematic arithmetic analysis on Phi-2.

1. Test all a+b for a,b in 1..14 — where does it work/fail?
2. How does the model represent numbers internally?
3. Is there a linear number line? Rotation? Something else?

Usage: .venv/Scripts/python.exe contrastive/code/arithmetic_deep.py
"""
import sys
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/phi-2"

print(f"Loading {MODEL}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
for p in model.parameters():
    p.requires_grad_(False)

NL = model.config.num_hidden_layers
W_U = model.lm_head.weight.detach()


def get_top1(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV))
    probs = torch.softmax(out.logits[0, -1].float(), -1)
    top1_id = int(probs.argmax())
    return tok.decode([top1_id]).strip(), float(probs[top1_id])


# ============================================================
# 1. Exhaustive addition test 1..14
# ============================================================
print("=" * 80)
print("1. ADDITION ACCURACY (a + b, a,b in 1..14)")
print("=" * 80)

grid = {}
for a in range(1, 15):
    for b in range(1, 15):
        expected = a + b
        top1, p = get_top1(f"{a} + {b} =")
        correct = top1 == str(expected)
        grid[(a, b)] = (expected, top1, correct, p)

# Print grid
print("\nAccuracy grid (. = correct, X = wrong):")
print("   b:", end="")
for b in range(1, 15):
    print(f" {b:>2}", end="")
print()
for a in range(1, 15):
    print(f"a={a:>2}:", end="")
    for b in range(1, 15):
        _, _, correct, _ = grid[(a, b)]
        print(f"  {'.' if correct else 'X'}", end="")
    print()

total = len(grid)
correct_count = sum(1 for v in grid.values() if v[2])
print(f"\nTotal: {correct_count}/{total} ({100*correct_count/total:.0f}%)")

# Show wrong answers
print("\nWrong answers:")
for (a, b), (exp, top1, correct, p) in sorted(grid.items()):
    if not correct:
        print(f"  {a} + {b} = {exp}, model says '{top1}' (P={p:.3f})")

# ============================================================
# 2. Number representation: fixed b, vary a
# ============================================================
print(f"\n{'='*80}")
print("2. NUMBER LINE REPRESENTATION")
print("=" * 80)

b_fixed = 3
print(f"\nFixed b={b_fixed}, varying a from 1 to 12")

states = {}
for a in range(1, 13):
    expr = f"{a} + {b_fixed} ="
    ids = tok(expr, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(
            torch.tensor([ids], device=DEV), output_hidden_states=True
        )
    states[a] = (
        out.hidden_states[28][0, -1, :].float().cpu().detach().clone()
    )
    top1, p = get_top1(expr)
    print(f"  {expr} -> {top1} (expected {a+b_fixed})")
    del out

# Pairwise cosines
print("\nPairwise cosine matrix (L28):")
print("     ", end="")
for a2 in range(1, 13):
    print(f" {a2:>4}", end="")
print()
for a1 in range(1, 13):
    print(f"  {a1:>2}:", end="")
    for a2 in range(1, 13):
        cos = float(
            torch.nn.functional.cosine_similarity(
                states[a1].unsqueeze(0), states[a2].unsqueeze(0)
            )
        )
        print(f" {cos:.2f}", end="")
    print()

# Step vectors
print("\nConsecutive step cos (does +1 always point the same way?):")
steps = []
for a in range(1, 12):
    d = states[a + 1] - states[a]
    steps.append(d)

for i in range(len(steps) - 1):
    cos = float(
        torch.nn.functional.cosine_similarity(
            steps[i].unsqueeze(0), steps[i + 1].unsqueeze(0)
        )
    )
    print(f"  step {i+1}->{i+2} vs step {i+2}->{i+3}: cos={cos:>+.3f}")

# Mean step
mean_step = torch.stack(steps).mean(dim=0)
mean_step_norm = mean_step / mean_step.norm()

# Project each state onto mean step — is it monotonic?
print("\nProjection onto mean step direction:")
projs = []
for a in range(1, 13):
    proj = float(states[a] @ mean_step_norm)
    projs.append(proj)
    print(f"  a={a:>2} (sum={a+b_fixed:>2}): proj={proj:>+8.1f}")

# Check monotonicity
mono = all(projs[i] < projs[i + 1] for i in range(len(projs) - 1))
print(f"  Monotonic: {mono}")

# SVD of the state matrix — what's the effective dimensionality?
mat = torch.stack([states[a] for a in range(1, 13)])
mat_centered = mat - mat.mean(dim=0)
U, S, V = torch.svd(mat_centered)
total_var = S.pow(2).sum()
print(f"\nSVD of centered states:")
for i in range(min(6, len(S))):
    pct = S[i] ** 2 / total_var
    print(f"  S{i+1} = {S[i]:.1f} ({pct:.1%})")

# Project into 2D and show arrangement
V2 = V[:, :2]
proj_2d = mat_centered @ V2
print("\n2D projection:")
for i, a in enumerate(range(1, 13)):
    x, y = float(proj_2d[i, 0]), float(proj_2d[i, 1])
    print(f"  a={a:>2} sum={a+b_fixed:>2}: ({x:>+7.1f}, {y:>+7.1f})")

# ============================================================
# 3. Does the model use a different representation for
#    different operations?
# ============================================================
print(f"\n{'='*80}")
print("3. OPERATION COMPARISON")
print("=" * 80)

ops = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
}

# Compare 5+3 vs 5-3 vs 5*3 at L28
for op_sym, op_fn in ops.items():
    expr = f"5 {op_sym} 3 ="
    expected = op_fn(5, 3)
    top1, p = get_top1(expr)
    print(f"  {expr} -> {top1} (expected {expected})")

# Get states for contrastive
op_states = {}
for op_sym in ["+", "-", "*"]:
    expr = f"5 {op_sym} 3 ="
    ids = tok(expr, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(
            torch.tensor([ids], device=DEV), output_hidden_states=True
        )
    op_states[op_sym] = (
        out.hidden_states[28][0, -1, :].float().cpu().detach().clone()
    )
    print(f"  Tokens for '{expr}': "
          f"{[tok.decode([t]).strip() for t in ids]}")
    del out

# Pairwise cosines
for o1 in ["+", "-", "*"]:
    for o2 in ["+", "-", "*"]:
        if o2 <= o1:
            continue
        cos = float(
            torch.nn.functional.cosine_similarity(
                op_states[o1].unsqueeze(0),
                op_states[o2].unsqueeze(0),
            )
        )
        print(f"  5{o1}3 vs 5{o2}3: cos={cos:.4f}")

# Contrastive: + vs -
dh = op_states["+"] - op_states["-"]
ld = dh @ W_U.float().cpu().T
v, i = torch.topk(ld, 6)
top = ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(6))
v, i = torch.topk(ld, 6, largest=False)
bot = ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(6))
print(f"\n  5+3 vs 5-3 contrast (L28):")
print(f"    + pole: [{top}]")
print(f"    - pole: [{bot}]")

# Contrastive: + vs *
dh = op_states["+"] - op_states["*"]
ld = dh @ W_U.float().cpu().T
v, i = torch.topk(ld, 6)
top = ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(6))
v, i = torch.topk(ld, 6, largest=False)
bot = ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(6))
print(f"\n  5+3 vs 5*3 contrast (L28):")
print(f"    + pole: [{top}]")
print(f"    * pole: [{bot}]")

torch.cuda.empty_cache()
print(f"\n{'='*80}")
print("DONE")
