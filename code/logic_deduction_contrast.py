"""
Contrastive analysis of logical deduction in Phi-2.

Four experiments:
1. Same premises, different question (shortest vs tallest)
2. Change one premise (answer flips)
3. Solvable vs unsolvable (connected vs disconnected chain)
4. Same structure, different names (control)

Usage: .venv/Scripts/python.exe contrastive/code/logic_deduction_contrast.py
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


def topk(logits, tok, k=6):
    v, i = torch.topk(logits, k)
    return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))


def botk(logits, tok, k=6):
    v, i = torch.topk(logits, k, largest=False)
    return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))


def run(model, tok, text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(
            torch.tensor([ids], device=DEV), output_hidden_states=True
        )
    return out, ids


def predict(model, tok, text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        gen = model.generate(
            torch.tensor([ids], device=DEV),
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(gen[0][len(ids) :]).strip().split("\n")[0][:60]


def contrast(out1, out2, label1, label2, desc, model, tok, NL, W_U):
    print(f"\n--- {desc} ---")
    for L in range(0, NL + 1, 4):
        h1 = out1.hidden_states[L][0, -1, :].float()
        h2 = out2.hidden_states[L][0, -1, :].float()
        dh = h1 - h2
        norm = float(dh.norm() / h1.norm())
        ld = dh @ W_U.float().T
        t = topk(ld, tok)
        b = botk(ld, tok)
        print(f"  L{L:>2} ({norm:.3f}) {label1}=[{t}]  {label2}=[{b}]")


def main():
    print(f"Loading {MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    for p in model.parameters():
        p.requires_grad_(False)

    NL = model.config.num_hidden_layers
    W_U = model.lm_head.weight.detach()

    # ================================================================
    # 1. SAME PREMISES, DIFFERENT QUESTION
    # ================================================================
    print("=" * 100)
    print("1. SAME PREMISES, DIFFERENT QUESTION")
    print("=" * 100)

    p1 = ("Alice is taller than Bob. Bob is taller than Carol. "
          "Who is the shortest?\nAnswer:")
    p2 = ("Alice is taller than Bob. Bob is taller than Carol. "
          "Who is the tallest?\nAnswer:")

    print(f'A: "{p1.replace(chr(10), " ")}"')
    print(f"  -> {predict(model, tok, p1)}")
    print(f'B: "{p2.replace(chr(10), " ")}"')
    print(f"  -> {predict(model, tok, p2)}")

    out1, _ = run(model, tok, p1)
    out2, _ = run(model, tok, p2)
    contrast(out1, out2, "shortest", "tallest",
             "shortest vs tallest (same premises)", model, tok, NL, W_U)
    del out1, out2
    torch.cuda.empty_cache()

    # ================================================================
    # 2. CHANGE ONE PREMISE — answer flips
    # ================================================================
    print(f"\n{'=' * 100}")
    print("2. CHANGE ONE PREMISE (answer flips)")
    print("=" * 100)

    # A>B>C shortest=Carol vs A>B, C>B shortest=Bob
    p_abc = ("Alice is taller than Bob. Bob is taller than Carol. "
             "Who is the shortest?\nAnswer:")
    p_acb = ("Alice is taller than Bob. Carol is taller than Bob. "
             "Who is the shortest?\nAnswer:")

    print(f'A: "{p_abc.replace(chr(10), " ")}"')
    print(f"  -> {predict(model, tok, p_abc)}")
    print(f'B: "{p_acb.replace(chr(10), " ")}"')
    print(f"  -> {predict(model, tok, p_acb)}")

    out_abc, _ = run(model, tok, p_abc)
    out_acb, _ = run(model, tok, p_acb)
    contrast(out_abc, out_acb, "Carol", "Bob",
             "B>C vs C>B (answer flips Carol<->Bob)", model, tok, NL, W_U)
    del out_abc, out_acb
    torch.cuda.empty_cache()

    # Also: flip first premise
    p_abc2 = ("Alice is taller than Bob. Bob is taller than Carol. "
              "Who is the shortest?\nAnswer:")
    p_bac = ("Bob is taller than Alice. Alice is taller than Carol. "
             "Who is the shortest?\nAnswer:")

    print(f'\nA: "{p_abc2.replace(chr(10), " ")}"')
    print(f"  -> {predict(model, tok, p_abc2)}")
    print(f'B: "{p_bac.replace(chr(10), " ")}"')
    print(f"  -> {predict(model, tok, p_bac)}")

    out_abc2, _ = run(model, tok, p_abc2)
    out_bac, _ = run(model, tok, p_bac)
    contrast(out_abc2, out_bac, "A>B", "B>A",
             "A>B vs B>A in first premise (tallest flips)", model, tok, NL, W_U)
    del out_abc2, out_bac
    torch.cuda.empty_cache()

    # ================================================================
    # 3. SOLVABLE VS UNSOLVABLE
    # ================================================================
    print(f"\n{'=' * 100}")
    print("3. SOLVABLE VS UNSOLVABLE")
    print("=" * 100)

    p_solv = ("Alice is taller than Bob. Bob is taller than Carol. "
              "Who is the shortest?\nAnswer:")
    p_unsolv = ("Alice is taller than Bob. Carol is taller than Dan. "
                "Who is the shortest?\nAnswer:")

    print(f'Solvable: "{p_solv.replace(chr(10), " ")}"')
    print(f"  -> {predict(model, tok, p_solv)}")
    print(f'Unsolvable: "{p_unsolv.replace(chr(10), " ")}"')
    print(f"  -> {predict(model, tok, p_unsolv)}")

    out_s, ids_s = run(model, tok, p_solv)
    out_u, ids_u = run(model, tok, p_unsolv)
    contrast(out_s, out_u, "solvable", "unsolvable",
             "connected chain vs disconnected", model, tok, NL, W_U)

    print(f"\nTokens solvable:   {[tok.decode([t]) for t in ids_s]}")
    print(f"Tokens unsolvable: {[tok.decode([t]) for t in ids_u]}")
    del out_s, out_u
    torch.cuda.empty_cache()

    # Also: solvable vs unsolvable with "can we determine" question
    p_meta_s = ("Alice is taller than Bob. Bob is taller than Carol. "
                "Can we determine who is the shortest?\nAnswer:")
    p_meta_u = ("Alice is taller than Bob. Carol is taller than Dan. "
                "Can we determine who is the shortest?\nAnswer:")

    print(f'\nMeta-solvable: "{p_meta_s.replace(chr(10), " ")}"')
    print(f"  -> {predict(model, tok, p_meta_s)}")
    print(f'Meta-unsolvable: "{p_meta_u.replace(chr(10), " ")}"')
    print(f"  -> {predict(model, tok, p_meta_u)}")

    out_ms, _ = run(model, tok, p_meta_s)
    out_mu, _ = run(model, tok, p_meta_u)
    contrast(out_ms, out_mu, "meta_solv", "meta_unsolv",
             "meta: can we determine (solvable vs not)", model, tok, NL, W_U)
    del out_ms, out_mu
    torch.cuda.empty_cache()

    # ================================================================
    # 4. SAME STRUCTURE, DIFFERENT NAMES
    # ================================================================
    print(f"\n{'=' * 100}")
    print("4. SAME STRUCTURE, DIFFERENT NAMES")
    print("=" * 100)

    p_abc3 = ("Alice is taller than Bob. Bob is taller than Carol. "
              "Who is the shortest?\nAnswer:")
    p_def = ("Dan is taller than Eve. Eve is taller than Frank. "
             "Who is the shortest?\nAnswer:")
    p_xyz = ("X is taller than Y. Y is taller than Z. "
             "Who is the shortest?\nAnswer:")

    for label, text in [("ABC", p_abc3), ("DEF", p_def), ("XYZ", p_xyz)]:
        print(f'{label}: "{text.replace(chr(10), " ")}"')
        print(f"  -> {predict(model, tok, text)}")

    out_abc3, _ = run(model, tok, p_abc3)
    out_def, _ = run(model, tok, p_def)
    out_xyz, _ = run(model, tok, p_xyz)

    contrast(out_abc3, out_def, "ABC", "DEF",
             "Alice/Bob/Carol vs Dan/Eve/Frank", model, tok, NL, W_U)
    contrast(out_abc3, out_xyz, "ABC", "XYZ",
             "Alice/Bob/Carol vs X/Y/Z", model, tok, NL, W_U)

    # Cross-name cosine at Answer position
    print("\n  Cross-name cosine at Answer position:")
    for L in range(0, NL + 1, 4):
        h_abc = out_abc3.hidden_states[L][0, -1, :].float()
        h_def = out_def.hidden_states[L][0, -1, :].float()
        h_xyz = out_xyz.hidden_states[L][0, -1, :].float()
        cos_ad = float(torch.nn.functional.cosine_similarity(
            h_abc.unsqueeze(0), h_def.unsqueeze(0)))
        cos_ax = float(torch.nn.functional.cosine_similarity(
            h_abc.unsqueeze(0), h_xyz.unsqueeze(0)))
        cos_dx = float(torch.nn.functional.cosine_similarity(
            h_def.unsqueeze(0), h_xyz.unsqueeze(0)))
        print(f"  L{L:>2}: ABC-DEF={cos_ad:.3f}  "
              f"ABC-XYZ={cos_ax:.3f}  DEF-XYZ={cos_dx:.3f}")

    del out_abc3, out_def, out_xyz
    torch.cuda.empty_cache()

    print(f"\n{'=' * 100}")
    print("DONE")


if __name__ == "__main__":
    main()
