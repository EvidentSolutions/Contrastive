"""
Epistemological cases: how does contrastive projection differ when the model
knows, doesn't know, hallucinates, or encounters false/changed facts?

1. Factual vs hallucination (model confidently wrong)
2. Known vs genuinely unknown (obscure facts)
3. True vs false statements
4. Temporal facts (things that changed)
5. Counterfactual framing
6. Confidence calibration — high-confidence vs low-confidence correct answers

Usage: .venv/Scripts/python.exe contrastive/code/epistemological.py
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
    return ", ".join(
        f"{tok.decode([int(i[j])]).strip()[:12]}" for j in range(k)
    )


def bk(logits, k=6):
    v, i = torch.topk(logits, k, largest=False)
    return ", ".join(
        f"{tok.decode([int(i[j])]).strip()[:12]}" for j in range(k)
    )


def predict(text, max_tokens=8):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        gen = model.generate(
            torch.tensor([ids], device=DEV),
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(gen[0][len(ids):]).strip().split("\n")[0][:40]


def run_contrastive(pa, pb, layers=None, label_a="A", label_b="B"):
    if layers is None:
        layers = _sl(8, 16, 20, 24, 28, 32)
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

    for L in layers:
        h_a = out_a.hidden_states[L][0, -1, :].float()
        h_b = out_b.hidden_states[L][0, -1, :].float()
        dh = h_a - h_b
        norm = float(dh.norm() / h_a.norm())
        ld = dh @ W_U.float().T
        print(f"    L{L:>2} ({norm:.3f}) {label_a}=[{tk(ld)}]  "
              f"{label_b}=[{bk(ld)}]")

    del out_a, out_b


def get_entropy(text):
    """Get prediction entropy (uncertainty measure)."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV))
    probs = torch.softmax(out.logits[0, -1].float(), -1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -float((probs * log_probs).sum())
    top5_p = float(probs.topk(5).values.sum())
    top1_tok = tok.decode([int(probs.argmax())]).strip()
    top1_p = float(probs.max())
    return entropy, top1_tok, top1_p, top5_p


# ============================================================
print("=" * 100)
print("1. FACTUAL vs HALLUCINATION")
print("   (Model knows A correctly, but what about B?)")
print("=" * 100)

# Well-known facts vs facts the model likely gets wrong
halluc_cases = [
    # (known, likely-wrong-or-hallucinated)
    ("The capital of France is",
     "The capital of Burkina Faso is"),
    ("The capital of Germany is",
     "The capital of Kyrgyzstan is"),
    ("The CEO of Apple is",
     "The CEO of Palantir is"),
    ("The inventor of the telephone is",
     "The inventor of the zipper is"),
    ("The chemical symbol for gold is",
     "The chemical symbol for rutherfordium is"),
    ("The tallest mountain in the world is",
     "The tallest mountain in Antarctica is"),
]

for pa, pb in halluc_cases:
    print()
    e_a, t_a, p_a, t5_a = get_entropy(pa)
    e_b, t_b, p_b, t5_b = get_entropy(pb)
    print(f'  Known:   "{pa}" -> {t_a} (P={p_a:.3f}, H={e_a:.2f})')
    print(f'  Obscure: "{pb}" -> {t_b} (P={p_b:.3f}, H={e_b:.2f})')
    run_contrastive(pa, pb, layers=_sl(16, 24, 28, 32),
                    label_a="Known", label_b="Obscure")

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("2. FACTUAL vs FICTIONAL (impossible to know)")
print("=" * 100)

fiction_cases = [
    ("The president of the United States is",
     "The president of Westeros is"),
    ("The population of Tokyo is",
     "The population of Rivendell is"),
    ("The currency of Japan is",
     "The currency of Narnia is"),
    ("Water boils at",
     "Mythril melts at"),
]

for pa, pb in fiction_cases:
    print()
    e_a, t_a, p_a, t5_a = get_entropy(pa)
    e_b, t_b, p_b, t5_b = get_entropy(pb)
    print(f'  Real:    "{pa}" -> {t_a} (P={p_a:.3f}, H={e_b:.2f})')
    print(f'  Fiction: "{pb}" -> {t_b} (P={p_b:.3f}, H={e_b:.2f})')
    run_contrastive(pa, pb, layers=_sl(16, 24, 28, 32),
                    label_a="Real", label_b="Fict")

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("3. TRUE vs FALSE STATEMENTS")
print("   (Does the model represent truth differently from falsehood?)")
print("=" * 100)

# Pairs where the model should know the correct answer
# We present true and false versions and see how the contrast reads
tf_pairs = [
    ("The Earth orbits the",
     "The Sun orbits the"),
    ("Paris is the capital of France. This is",
     "Paris is the capital of Germany. This is"),
    ("Water freezes at 0 degrees. This fact is",
     "Water freezes at 50 degrees. This fact is"),
    ("Humans have 206 bones. The number of bones is",
     "Humans have 500 bones. The number of bones is"),
    ("The speed of light is faster than the speed of sound. This is",
     "The speed of sound is faster than the speed of light. This is"),
]

for pa, pb in tf_pairs:
    print()
    run_contrastive(pa, pb, layers=_sl(16, 24, 28, 32),
                    label_a="True", label_b="False")

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("4. TEMPORAL FACTS (changed over time)")
print("   (Model trained ~2023, so some facts have changed)")
print("=" * 100)

temporal = [
    # Stable vs changed
    ("The capital of Australia is",
     "The capital of Kazakhstan is"),  # Astana->Nur-Sultan->Astana
    ("The queen of England is",
     "The king of England is"),
    ("The president of Russia is",
     "The president of France is"),
]

for pa, pb in temporal:
    print()
    e_a, t_a, p_a, _ = get_entropy(pa)
    e_b, t_b, p_b, _ = get_entropy(pb)
    print(f'  "{pa}" -> {t_a} (P={p_a:.3f}, H={e_a:.2f})')
    print(f'  "{pb}" -> {t_b} (P={p_b:.3f}, H={e_b:.2f})')
    run_contrastive(pa, pb, layers=_sl(16, 24, 28, 32))

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("5. COUNTERFACTUAL FRAMING")
print("   (Does 'imagine that' change the representation?)")
print("=" * 100)

counter = [
    ("The capital of France is",
     "Imagine that the capital of France is"),
    ("The Eiffel Tower is in",
     "In an alternate universe, the Eiffel Tower is in"),
    ("Einstein was born in",
     "If Einstein had been born in"),
]

for pa, pb in counter:
    print()
    run_contrastive(pa, pb, layers=_sl(16, 24, 28, 32),
                    label_a="Factual", label_b="Counter")

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("6. CONFIDENCE CALIBRATION")
print("   (High-confidence correct vs low-confidence correct)")
print("=" * 100)

# Compare entropy/confidence across many factual prompts
calibration_prompts = [
    "The capital of France is",
    "The capital of Italy is",
    "The capital of Spain is",
    "The capital of Poland is",
    "The capital of Norway is",
    "The capital of Portugal is",
    "The capital of Finland is",
    "The capital of Croatia is",
    "The capital of Latvia is",
    "The capital of Moldova is",
    "The capital of Bhutan is",
    "The capital of Lesotho is",
    "The capital of Suriname is",
    "The capital of Djibouti is",
    "The capital of Comoros is",
    "The capital of Tuvalu is",
]

# Ground truth
ground_truth = {
    "France": "Paris", "Italy": "Rome", "Spain": "Madrid",
    "Poland": "Warsaw", "Norway": "Oslo", "Portugal": "Lisbon",
    "Finland": "Helsinki", "Croatia": "Zagreb", "Latvia": "Riga",
    "Moldova": "Chisinau", "Bhutan": "Thimphu", "Lesotho": "Maseru",
    "Suriname": "Paramaribo", "Djibouti": "Djibouti",
    "Comoros": "Moroni", "Tuvalu": "Funafuti",
}

print(f"\n  {'Prompt':<40} {'Pred':<15} {'Correct?':<8} "
      f"{'P(top1)':<8} {'H':<6} {'top5%'}")
for prompt in calibration_prompts:
    entropy, top1, p1, t5 = get_entropy(prompt)
    # Extract country name
    country = prompt.split("of ")[-1].rstrip(" is")
    correct_ans = ground_truth.get(country, "?")
    is_correct = correct_ans.lower() in top1.lower() or top1.lower() in correct_ans.lower()
    print(f"  {prompt:<40} {top1:<15} {'Y' if is_correct else 'N':^8} "
          f"{p1:<8.3f} {entropy:<6.2f} {t5:.3f}")

# Contrastive: high-confidence vs low-confidence capital
print("\n  High-confidence (France) vs low-confidence capital:")
run_contrastive(
    "The capital of France is",
    "The capital of Comoros is",
    layers=_sl(16, 24, 28, 32),
    label_a="HiConf", label_b="LoConf"
)

# Two low-confidence ones against each other
print("\n  Two low-confidence capitals:")
run_contrastive(
    "The capital of Tuvalu is",
    "The capital of Comoros is",
    layers=_sl(16, 24, 28, 32),
    label_a="Tuvalu", label_b="Comoros"
)

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("7. HEDGED vs COMMITTED RECALL")
print("   (Same fact, different framing confidence)")
print("=" * 100)

hedge_cases = [
    ("The capital of France is definitely",
     "The capital of France is probably"),
    ("I am certain that water boils at",
     "I think water boils at"),
    ("Everyone knows that Einstein invented",
     "Some people believe that Einstein invented"),
]

for pa, pb in hedge_cases:
    print()
    run_contrastive(pa, pb, layers=_sl(16, 24, 28, 32),
                    label_a="Committed", label_b="Hedged")

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("8. CORRECT vs PRIMED-WRONG")
print("   (Can we see the model resisting a false prime?)")
print("=" * 100)

primed = [
    ("The capital of France is",
     "Berlin is a city in Germany. The capital of France is"),
    ("2 + 2 =",
     "3 + 3 = 6. 2 + 2 ="),
    ("Water boils at 100 degrees. At what temperature does water freeze?",
     "Water boils at 200 degrees. At what temperature does water freeze?"),
]

for pa, pb in primed:
    print()
    e_a, t_a, p_a, _ = get_entropy(pa)
    e_b, t_b, p_b, _ = get_entropy(pb)
    print(f'  Plain:  "{pa}" -> {t_a} (P={p_a:.3f})')
    print(f'  Primed: "{pb}" -> {t_b} (P={p_b:.3f})')
    # These have different lengths, so full contrastive doesn't apply
    # Just compare predictions and entropy

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("9. ENTROPY LANDSCAPE — same structure, different knowledge")
print("=" * 100)

# Compare hidden state norms and entropy for known vs unknown
known_unknown = [
    ("The Eiffel Tower is in", "known"),
    ("The Crystal Palace of Zoria is in", "fictional"),
    ("The capital of France is", "well-known"),
    ("The capital of Narnia is", "fictional"),
    ("Shakespeare wrote", "well-known"),
    ("Zaldor the Wise wrote", "fictional"),
    ("Einstein developed", "well-known"),
    ("Glorb developed", "fictional"),
]

print(f"\n  {'Prompt':<42} {'Type':<12} {'Pred':<15} "
      f"{'P(top1)':<8} {'H':<6}")
for prompt, ptype in known_unknown:
    entropy, top1, p1, _ = get_entropy(prompt)
    print(f"  {prompt:<42} {ptype:<12} {top1:<15} {p1:<8.3f} {entropy:<6.2f}")

# Hidden state norms across layers for known vs fictional
print("\n  Hidden state norms at last position:")
print(f"  {'Prompt':<42} ", end="")
for L in _sl(0, 8, 16, 24, 28, 32):
    print(f" L{L:>2}", end="")
print()

for prompt, ptype in known_unknown:
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(
            torch.tensor([ids], device=DEV), output_hidden_states=True
        )
    print(f"  {prompt:<42} ", end="")
    for L in _sl(0, 8, 16, 24, 28, 32):
        h = out.hidden_states[L][0, -1, :].float()
        print(f" {float(h.norm()):>3.0f}", end="")
    print(f"  ({ptype})")
    del out

torch.cuda.empty_cache()
print(f"\n{'='*100}")
print("DONE")
