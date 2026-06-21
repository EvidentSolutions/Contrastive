"""
Does the epistemic verb change what the model retrieves?
"I doubt that the capital of France is" -> Berlin (not Paris!)

Systematic test across domains and epistemic verbs.

Usage: .venv/Scripts/python.exe contrastive/code/epistemic_verbs.py
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


def get_top_probs(text, k=5):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV))
    probs = torch.softmax(out.logits[0, -1].float(), -1)
    v, i = torch.topk(probs, k)
    return [(tok.decode([int(i[j])]).strip(), float(v[j])) for j in range(k)]


def tk(logits, k=5):
    v, i = torch.topk(logits, k)
    return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))


def bk(logits, k=5):
    v, i = torch.topk(logits, k, largest=False)
    return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))


# ============================================================
print("=" * 100)
print("1. EPISTEMIC VERB SURVEY — capitals")
print("=" * 100)

verbs = [
    ("bare",     "The capital of {} is"),
    ("know",     "I know that the capital of {} is"),
    ("believe",  "I believe that the capital of {} is"),
    ("think",    "I think the capital of {} is"),
    ("suspect",  "I suspect that the capital of {} is"),
    ("doubt",    "I doubt that the capital of {} is"),
    ("deny",     "I deny that the capital of {} is"),
    ("unsure",   "I am unsure whether the capital of {} is"),
    ("certain",  "I am certain that the capital of {} is"),
    ("false",    "It is false that the capital of {} is"),
    ("unlikely", "It is unlikely that the capital of {} is"),
]

countries = ["France", "Germany", "Japan", "Italy", "Spain",
             "Australia", "Brazil", "Egypt"]

correct = {
    "France": "Paris", "Germany": "Berlin", "Japan": "Tokyo",
    "Italy": "Rome", "Spain": "Madrid", "Australia": "Canberra",
    "Brazil": "Brasilia", "Egypt": "Cairo",
}

print(f"\n  {'Verb':<10}", end="")
for c in countries:
    print(f" {c[:5]:>7}", end="")
print()

for verb_name, template in verbs:
    print(f"  {verb_name:<10}", end="")
    for country in countries:
        prompt = template.format(country)
        top = get_top_probs(prompt, 1)
        pred = top[0][0][:7]
        # Check if correct
        is_correct = correct[country].lower() in pred.lower() or \
                     pred.lower() in correct[country].lower()
        mark = "" if is_correct else "*"
        print(f" {pred+mark:>7}", end="")
    print()

print("\n  * = different from bare prediction")

# ============================================================
print(f"\n{'='*100}")
print("2. DETAILED: France with all verbs")
print("=" * 100)

print(f"\n  {'Verb':<10} {'Prediction':<25} {'Top-5 probs'}")
for verb_name, template in verbs:
    prompt = template.format("France")
    ans = predict(prompt, 8)
    top5 = get_top_probs(prompt, 5)
    top5_str = ", ".join(f"{t}({p:.2f})" for t, p in top5)
    print(f"  {verb_name:<10} {ans:<25} [{top5_str}]")

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("3. EPISTEMIC VERBS — other domains")
print("=" * 100)

domains = [
    ("Shakespeare", [
        ("bare",    "Shakespeare wrote"),
        ("know",    "I know that Shakespeare wrote"),
        ("doubt",   "I doubt that Shakespeare wrote"),
        ("deny",    "I deny that Shakespeare wrote"),
        ("false",   "It is false that Shakespeare wrote"),
    ]),
    ("Boiling point", [
        ("bare",    "Water boils at"),
        ("know",    "I know that water boils at"),
        ("doubt",   "I doubt that water boils at"),
        ("false",   "It is false that water boils at"),
    ]),
    ("Einstein", [
        ("bare",    "Einstein was born in"),
        ("know",    "I know that Einstein was born in"),
        ("doubt",   "I doubt that Einstein was born in"),
        ("false",   "It is false that Einstein was born in"),
    ]),
    ("Oxygen", [
        ("bare",    "Humans breathe"),
        ("know",    "I know that humans breathe"),
        ("doubt",   "I doubt that humans breathe"),
        ("false",   "It is false that humans breathe"),
    ]),
]

for domain_name, cases in domains:
    print(f"\n  --- {domain_name} ---")
    for verb_name, prompt in cases:
        ans = predict(prompt, 10)
        top3 = get_top_probs(prompt, 3)
        top3_str = ", ".join(f"{t}({p:.2f})" for t, p in top3)
        print(f'  {verb_name:<8} "{prompt}" -> "{ans}"')
        print(f"           Top3: [{top3_str}]")

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("4. CONTRASTIVE: know vs doubt")
print("=" * 100)

contrast_cases = [
    ("I know that the capital of France is",
     "I doubt that the capital of France is"),
    ("I know that the capital of Germany is",
     "I doubt that the capital of Germany is"),
    ("I know that the capital of Japan is",
     "I doubt that the capital of Japan is"),
    ("I know that Shakespeare wrote",
     "I doubt that Shakespeare wrote"),
    ("I know that water boils at",
     "I doubt that water boils at"),
    ("I know that Einstein was born in",
     "I doubt that Einstein was born in"),
]

for pa, pb in contrast_cases:
    ids_a = tok(pa, add_special_tokens=False)["input_ids"]
    ids_b = tok(pb, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out_a = model(
            torch.tensor([ids_a], device=DEV), output_hidden_states=True
        )
        out_b = model(
            torch.tensor([ids_b], device=DEV), output_hidden_states=True
        )
    ans_a = predict(pa, 5)
    ans_b = predict(pb, 5)
    print(f'\n  Know:  "{pa}" -> "{ans_a}"')
    print(f'  Doubt: "{pb}" -> "{ans_b}"')
    for L in _sl(20, 24, 28, 32):
        h_a = out_a.hidden_states[L][0, -1, :].float()
        h_b = out_b.hidden_states[L][0, -1, :].float()
        dh = h_a - h_b
        norm = float(dh.norm() / h_a.norm())
        ld = dh @ W_U.float().T
        print(f"    L{L:>2} ({norm:.3f}) Know=[{tk(ld)}]  Doubt=[{bk(ld)}]")
    del out_a, out_b

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("5. CONTRASTIVE: certain vs unsure")
print("=" * 100)

certain_unsure = [
    ("I am certain that the capital of France is",
     "I am unsure whether the capital of France is"),
    ("I am certain that water boils at",
     "I am unsure whether water boils at"),
    ("I am certain that Shakespeare wrote",
     "I am unsure whether Shakespeare wrote"),
]

for pa, pb in certain_unsure:
    ids_a = tok(pa, add_special_tokens=False)["input_ids"]
    ids_b = tok(pb, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out_a = model(
            torch.tensor([ids_a], device=DEV), output_hidden_states=True
        )
        out_b = model(
            torch.tensor([ids_b], device=DEV), output_hidden_states=True
        )
    ans_a = predict(pa, 8)
    ans_b = predict(pb, 8)
    print(f'\n  Certain: "{pa}" -> "{ans_a}"')
    print(f'  Unsure:  "{pb}" -> "{ans_b}"')
    for L in _sl(24, 28, 32):
        h_a = out_a.hidden_states[L][0, -1, :].float()
        h_b = out_b.hidden_states[L][0, -1, :].float()
        dh = h_a - h_b
        ld = dh @ W_U.float().T
        print(f"    L{L:>2} Cert=[{tk(ld)}]  Unsr=[{bk(ld)}]")
    del out_a, out_b

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("6. IS 'DOUBT -> BERLIN' REAL OR JUST TOKEN PRIMING?")
print("   Control: does any negative word before a fact change recall?")
print("=" * 100)

priming_controls = [
    "The capital of France is",
    "I doubt that the capital of France is",
    "Sadly, the capital of France is",
    "Unfortunately, the capital of France is",
    "Surprisingly, the capital of France is",
    "Allegedly, the capital of France is",
    "Supposedly, the capital of France is",
    "Incorrectly, the capital of France is",
    "The wrong capital of France is",
    "The fake capital of France is",
    "Not the capital of France is",
]

print(f"\n  {'Prompt':<55} {'Top-3'}")
for prompt in priming_controls:
    top3 = get_top_probs(prompt, 3)
    top3_str = ", ".join(f"{t}({p:.2f})" for t, p in top3)
    ans = predict(prompt, 5)
    print(f'  "{prompt}"')
    print(f'    -> "{ans}"  [{top3_str}]')

torch.cuda.empty_cache()
print(f"\n{'='*100}")
print("DONE")
