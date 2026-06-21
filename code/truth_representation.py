"""
Focused study of truth/falsehood representation in Phi-2.

1. Does the model consistently represent true vs false in contrastive space?
2. Is there a stable "truth direction" that transfers across domains?
3. Negation: does "not" flip the truth direction?
4. Near-miss: "Paris is the capital of Spain" — how does it differ from
   clearly-false "Paris is the capital of 7"?
5. Self-contradiction: conflicting premises

Usage: .venv/Scripts/python.exe contrastive/code/truth_representation.py
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

print(f"Loading {MODEL}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
for p in model.parameters():
    p.requires_grad_(False)

NL = model.config.num_hidden_layers
W_U = model.lm_head.weight.detach()


def _sl(*layers):
    """Scale layer indices from 32-layer base to current NL."""
    return sorted(set(min(round(l * NL / 32), NL) for l in layers))


def tk(logits, k=6):
    v, i = torch.topk(logits, k)
    return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))


def bk(logits, k=6):
    v, i = torch.topk(logits, k, largest=False)
    return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))


def predict(text, max_tokens=10):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        gen = model.generate(
            torch.tensor([ids], device=DEV),
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(gen[0][len(ids):]).strip().split("\n")[0][:50]


def get_hidden(text, layer=None):
    if layer is None:
        layer = _sl(28)[0]
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(
            torch.tensor([ids], device=DEV), output_hidden_states=True
        )
    h = out.hidden_states[layer][0, -1, :].float().cpu().detach().clone()
    del out
    return h


def contrast_pair(pa, pb, layers=None, label_a="T", label_b="F"):
    if layers is None:
        layers = _sl(16, 24, 28, 32)
    ids_a = tok(pa, add_special_tokens=False)["input_ids"]
    ids_b = tok(pb, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out_a = model(
            torch.tensor([ids_a], device=DEV), output_hidden_states=True
        )
        out_b = model(
            torch.tensor([ids_b], device=DEV), output_hidden_states=True
        )
    ans_a = predict(pa)
    ans_b = predict(pb)
    print(f'  {label_a}: "{pa}" -> "{ans_a}"')
    print(f'  {label_b}: "{pb}" -> "{ans_b}"')
    dh_by_layer = {}
    for L in layers:
        h_a = out_a.hidden_states[L][0, -1, :].float()
        h_b = out_b.hidden_states[L][0, -1, :].float()
        dh = h_a - h_b
        dh_by_layer[L] = dh.cpu().detach().clone()
        norm = float(dh.norm() / h_a.norm())
        ld = dh @ W_U.float().T
        print(f"    L{L:>2} ({norm:.3f}) {label_a}=[{tk(ld)}]  "
              f"{label_b}=[{bk(ld)}]")
    del out_a, out_b
    return dh_by_layer


# ============================================================
print("=" * 100)
print("1. TRUTH DIRECTION — consistency across domains")
print("=" * 100)

# Extract truth direction from many domains at L28
truth_pairs = [
    ("Paris is the capital of France. This is",
     "Paris is the capital of Germany. This is"),
    ("Water freezes at 0 degrees. This fact is",
     "Water freezes at 50 degrees. This fact is"),
    ("The speed of light is faster than the speed of sound. This is",
     "The speed of sound is faster than the speed of light. This is"),
    ("Humans breathe oxygen. This statement is",
     "Humans breathe nitrogen. This statement is"),
    ("The Sun is a star. This claim is",
     "The Sun is a planet. This claim is"),
    ("Dogs are mammals. This is",
     "Dogs are reptiles. This is"),
    ("Tokyo is in Japan. This is",
     "Tokyo is in Brazil. This is"),
    ("Iron is a metal. This is",
     "Iron is a gas. This is"),
]

# Compute truth directions at L28
truth_directions = []
for pa, pb in truth_pairs:
    print()
    dh = contrast_pair(pa, pb, layers=_sl(28))
    truth_directions.append(dh[_sl(28)[0]])

# Pairwise cosines between truth directions
print(f"\n  Pairwise cosine between truth directions (L{_sl(28)[0]}):")
labels = ["Paris", "Water", "Light", "Oxygen", "Star", "Mammal",
          "Tokyo", "Metal"]
print("         ", end="")
for l in labels:
    print(f" {l[:5]:>5}", end="")
print()
for i in range(len(truth_directions)):
    print(f"  {labels[i][:5]:>5}:", end="")
    for j in range(len(truth_directions)):
        cos = float(torch.nn.functional.cosine_similarity(
            truth_directions[i].unsqueeze(0),
            truth_directions[j].unsqueeze(0),
        ))
        print(f" {cos:>+.3f}", end="")
    print()

# Mean truth direction
mean_truth = torch.stack(truth_directions).mean(dim=0)
mean_truth_norm = mean_truth / mean_truth.norm()
print(f"\n  Mean truth direction reads as:")
ld = mean_truth @ W_U.float().cpu().T
print(f"    True pole: [{tk(ld)}]")
print(f"    False pole: [{bk(ld)}]")

# Does the mean direction transfer to held-out cases?
print(f"\n  Transfer to held-out cases:")
held_out = [
    ("Diamonds are hard. This is",
     "Diamonds are soft. This is"),
    ("The Atlantic Ocean is larger than the Pacific. This is",
     "The Pacific Ocean is larger than the Atlantic. This is"),
    ("Shakespeare was English. This is",
     "Shakespeare was Japanese. This is"),
    ("Gravity pulls objects downward. This is",
     "Gravity pushes objects upward. This is"),
]

for pa, pb in held_out:
    h_a = get_hidden(pa, _sl(28)[0])
    h_b = get_hidden(pb, _sl(28)[0])
    dh = h_a - h_b
    # Project onto mean truth direction
    proj = float(dh @ mean_truth_norm)
    # Also check what it reads
    ld = dh @ W_U.float().cpu().T
    ans_a = predict(pa, 5)
    ans_b = predict(pb, 5)
    print(f'  "{pa[:50]}" -> "{ans_a}"')
    print(f'  "{pb[:50]}" -> "{ans_b}"')
    print(f"    Proj onto truth dir: {proj:>+.1f}  "
          f"T=[{tk(ld, 3)}]  F=[{bk(ld, 3)}]")

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("2. NEGATION — does 'not' flip the truth direction?")
print("=" * 100)

negation_cases = [
    ("Paris is the capital of France. This is",
     "Paris is not the capital of France. This is"),
    ("Water freezes at 0 degrees. This is",
     "Water does not freeze at 0 degrees. This is"),
    ("The Sun is a star. This is",
     "The Sun is not a star. This is"),
    ("Dogs are mammals. This is",
     "Dogs are not mammals. This is"),
]

for pa, pb in negation_cases:
    print()
    dh = contrast_pair(pa, pb, layers=_sl(28),
                       label_a="Affirm", label_b="Negate")
    # Project onto mean truth direction
    proj = float(dh[_sl(28)[0]] @ mean_truth_norm)
    print(f"    Proj onto truth dir: {proj:>+.1f}")

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("3. DEGREES OF FALSEHOOD")
print("   (Clearly false, near-miss, absurd)")
print("=" * 100)

degrees = [
    ("Paris is the capital of France. This is",      "true"),
    ("Paris is the capital of Belgium. This is",     "near-miss"),
    ("Paris is the capital of Japan. This is",       "wrong-continent"),
    ("Paris is the capital of 7. This is",           "absurd"),
    ("Paris is the capital of happiness. This is",   "category-error"),
]

print("\n  Projections onto truth direction and predictions:")
base_h = get_hidden(degrees[0][0], _sl(28)[0])
for text, label in degrees:
    h = get_hidden(text, _sl(28)[0])
    dh = h - base_h
    proj = float(h @ mean_truth_norm)
    ans = predict(text, 5)
    ld = dh @ W_U.float().cpu().T if label != "true" else torch.zeros(W_U.shape[0])
    print(f'  [{label:<16}] "{text}" -> "{ans}"')
    print(f"    Proj: {proj:>+.1f}")
    if label != "true":
        print(f"    Contrast vs true: T=[{tk(ld, 3)}]  F=[{bk(ld, 3)}]")

# Direct pairwise — near-miss vs absurd
print("\n  Near-miss vs absurd (both false, different degree):")
contrast_pair(
    "Paris is the capital of Belgium. This is",
    "Paris is the capital of 7. This is",
    layers=_sl(24, 28, 32),
    label_a="Near", label_b="Absurd"
)

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("4. SELF-CONTRADICTION")
print("=" * 100)

contra = [
    ("Paris is the capital of France. Paris is the capital of France. Paris is",
     "Paris is the capital of France. Paris is the capital of Germany. Paris is"),
    ("Water boils at 100 degrees. Water boils at 100 degrees. Water boils at",
     "Water boils at 100 degrees. Water boils at 200 degrees. Water boils at"),
    ("The sky is blue. The sky is blue. The color of the sky is",
     "The sky is blue. The sky is green. The color of the sky is"),
]

for pa, pb in contra:
    print()
    contrast_pair(pa, pb, layers=_sl(24, 28, 32),
                  label_a="Consist", label_b="Contra")

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("5. KNOWLEDGE BOUNDARIES")
print("   (Known true, known false, genuinely unknown)")
print("=" * 100)

# Compare: model-knows-true, model-knows-false, model-doesn't-know
boundary = [
    ("The capital of France is Paris. This is",         "known-true"),
    ("The capital of France is Berlin. This is",        "known-false"),
    ("The capital of Tuvalu is Funafuti. This is",      "obscure-true"),
    ("The capital of Tuvalu is Moroni. This is",        "obscure-false"),
]

print("\n  Predictions and truth-direction projections:")
for text, label in boundary:
    h = get_hidden(text, _sl(28)[0])
    proj = float(h @ mean_truth_norm)
    ans = predict(text, 5)
    print(f'  [{label:<14}] "{text}" -> "{ans}"  proj={proj:>+.1f}')

# Pairwise contrasts
print("\n  Known-true vs known-false:")
contrast_pair(boundary[0][0], boundary[1][0], layers=_sl(28),
              label_a="KnTrue", label_b="KnFalse")

print("\n  Obscure-true vs obscure-false:")
contrast_pair(boundary[2][0], boundary[3][0], layers=_sl(28),
              label_a="ObTrue", label_b="ObFalse")

print("\n  Known-true vs obscure-true:")
contrast_pair(boundary[0][0], boundary[2][0], layers=_sl(28),
              label_a="KnTrue", label_b="ObTrue")

print("\n  Known-false vs obscure-false:")
contrast_pair(boundary[1][0], boundary[3][0], layers=_sl(28),
              label_a="KnFalse", label_b="ObFalse")

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("6. EPISTEMIC VERBS")
print("   (know, believe, suspect, doubt)")
print("=" * 100)

epistemic = [
    "I know that the capital of France is",
    "I believe that the capital of France is",
    "I suspect that the capital of France is",
    "I doubt that the capital of France is",
    "I am certain that the capital of France is",
    "I am unsure whether the capital of France is",
]

print("\n  Predictions:")
for prompt in epistemic:
    ans = predict(prompt, 5)
    print(f'  "{prompt}" -> "{ans}"')

# Pairwise contrasts: know vs doubt, certain vs unsure
print("\n  Know vs doubt:")
contrast_pair(epistemic[0], epistemic[3], layers=_sl(24, 28, 32),
              label_a="Know", label_b="Doubt")

print("\n  Certain vs unsure:")
contrast_pair(epistemic[4], epistemic[5], layers=_sl(24, 28, 32),
              label_a="Certain", label_b="Unsure")

print("\n  Know vs believe:")
contrast_pair(epistemic[0], epistemic[1], layers=_sl(24, 28, 32),
              label_a="Know", label_b="Believe")

torch.cuda.empty_cache()
print(f"\n{'='*100}")
print("DONE")
