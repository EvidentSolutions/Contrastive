"""
Geometric analysis of ordering representation.

How are name tokens arranged in the residual stream space?
Is the ordering a rotation? A linear projection? Something else?

Usage: .venv/Scripts/python.exe contrastive/code/geometric_ordering.py
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

    def _sl(*layers):
        """Scale layer indices from 32-layer base to current NL."""
        return sorted(set(min(round(l * NL / 32), NL) for l in layers))

    prompts = {
        "A>B>C": "Alice is taller than Bob. Bob is taller than Carol.",
        "A>C>B": "Alice is taller than Carol. Carol is taller than Bob.",
        "B>A>C": "Bob is taller than Alice. Alice is taller than Carol.",
        "B>C>A": "Bob is taller than Carol. Carol is taller than Alice.",
        "C>A>B": "Carol is taller than Alice. Alice is taller than Bob.",
        "C>B>A": "Carol is taller than Bob. Bob is taller than Alice.",
    }

    alice_id = tok(" Alice", add_special_tokens=False)["input_ids"][0]
    bob_id = tok(" Bob", add_special_tokens=False)["input_ids"][0]
    carol_id = tok(" Carol", add_special_tokens=False)["input_ids"][0]

    w_alice = W_U[alice_id].float().cpu()
    w_bob = W_U[bob_id].float().cpu()
    w_carol = W_U[carol_id].float().cpu()

    for L in _sl(16, 24, 28, 32):
        print(f"\n{'='*100}")
        print(f"Layer {L}")
        print(f"{'='*100}")

        logit_vecs = {}
        for key, text in prompts.items():
            ids = tok(text, add_special_tokens=False)["input_ids"]
            period_pos = max(
                i for i, t in enumerate(ids) if tok.decode([t]).strip() == "."
            )
            with torch.no_grad():
                out = model(
                    torch.tensor([ids], device=DEV),
                    output_hidden_states=True,
                )
            h = out.hidden_states[L][0, period_pos, :].float().cpu()
            la = float(h @ w_alice)
            lb = float(h @ w_bob)
            lc = float(h @ w_carol)
            logit_vecs[key] = torch.tensor([la, lb, lc])
            del out

        # Show raw logits
        print("\n  Raw name logits (Alice, Bob, Carol):")
        for key, v in logit_vecs.items():
            order = key.replace(">", "")
            top, mid, bot = order[0], order[1], order[2]
            print(
                f"    {key:>8}: A={v[0]:>+7.1f} B={v[1]:>+7.1f} C={v[2]:>+7.1f}"
                f"  | top={top} mid={mid} bot={bot}"
            )

        # By ROLE: extract logit for top/mid/bottom entity
        print("\n  Logits by ROLE (not name):")
        name_idx = {"A": 0, "B": 1, "C": 2}
        role_logits = {"top": [], "mid": [], "bot": []}
        for key in prompts:
            order = key.replace(">", "")
            v = logit_vecs[key]
            for role, pos in [("top", 0), ("mid", 1), ("bot", 2)]:
                role_logits[role].append(float(v[name_idx[order[pos]]]))

        for role in ["top", "mid", "bot"]:
            vals = role_logits[role]
            mean = sum(vals) / len(vals)
            std = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
            print(
                f"    {role}: mean={mean:>+7.1f} std={std:>5.1f}"
                f"  vals={[round(v, 1) for v in vals]}"
            )

        # Center the logit vectors (subtract mean)
        all_vecs = torch.stack(list(logit_vecs.values()))
        center = all_vecs.mean(dim=0)
        centered = {k: v - center for k, v in logit_vecs.items()}

        # Pairwise angles between centered vectors
        print("\n  Pairwise angles (centered):")
        keys = list(centered.keys())
        for i, k1 in enumerate(keys):
            for k2 in keys[i + 1 :]:
                v1 = centered[k1]
                v2 = centered[k2]
                cos = float(
                    torch.nn.functional.cosine_similarity(
                        v1.unsqueeze(0), v2.unsqueeze(0)
                    )
                )
                angle = float(
                    torch.acos(torch.clamp(torch.tensor(cos), -1, 1))
                    * 180
                    / 3.14159
                )
                # Same bottom?
                b1 = k1.replace(">", "")[-1]
                b2 = k2.replace(">", "")[-1]
                same_b = "*" if b1 == b2 else " "
                # Cyclic?
                cyc1 = k1.replace(">", "") in ["ABC", "BCA", "CAB"]
                cyc2 = k2.replace(">", "") in ["ABC", "BCA", "CAB"]
                same_cyc = "c" if cyc1 == cyc2 else " "
                print(
                    f"    {k1:>8} vs {k2:<8}:"
                    f" angle={angle:>6.1f}deg  cos={cos:>+.3f}"
                    f"  bot={b1}/{b2}{same_b} {same_cyc}"
                )

        # Check rotation: cyclic group (ABC, BCA, CAB) should be 120deg apart
        # anti-cyclic (ACB, CBA, BAC) should be the other 120deg set
        cyclic = ["A>B>C", "B>C>A", "C>A>B"]
        anticyclic = ["A>C>B", "C>B>A", "B>A>C"]

        print("\n  Cyclic group (A>B>C, B>C>A, C>A>B) angles:")
        for i in range(3):
            for j in range(i + 1, 3):
                v1 = centered[cyclic[i]]
                v2 = centered[cyclic[j]]
                cos = float(
                    torch.nn.functional.cosine_similarity(
                        v1.unsqueeze(0), v2.unsqueeze(0)
                    )
                )
                angle = float(
                    torch.acos(torch.clamp(torch.tensor(cos), -1, 1))
                    * 180
                    / 3.14159
                )
                print(
                    f"    {cyclic[i]:>8} vs {cyclic[j]:<8}:"
                    f" angle={angle:>6.1f}deg (120deg = rotation)"
                )

        print("  Anti-cyclic group (A>C>B, C>B>A, B>A>C) angles:")
        for i in range(3):
            for j in range(i + 1, 3):
                v1 = centered[anticyclic[i]]
                v2 = centered[anticyclic[j]]
                cos = float(
                    torch.nn.functional.cosine_similarity(
                        v1.unsqueeze(0), v2.unsqueeze(0)
                    )
                )
                angle = float(
                    torch.acos(torch.clamp(torch.tensor(cos), -1, 1))
                    * 180
                    / 3.14159
                )
                print(
                    f"    {anticyclic[i]:>8} vs {anticyclic[j]:<8}:"
                    f" angle={angle:>6.1f}deg"
                )

        # Cross group: cyclic vs anticyclic
        print("  Cross (cyclic vs anti-cyclic):")
        for c in cyclic:
            for a in anticyclic:
                v1 = centered[c]
                v2 = centered[a]
                cos = float(
                    torch.nn.functional.cosine_similarity(
                        v1.unsqueeze(0), v2.unsqueeze(0)
                    )
                )
                angle = float(
                    torch.acos(torch.clamp(torch.tensor(cos), -1, 1))
                    * 180
                    / 3.14159
                )
                print(
                    f"    {c:>8} vs {a:<8}:"
                    f" angle={angle:>6.1f}deg"
                )

        # SVD of the 6x3 centered matrix — what's the effective dimensionality?
        mat = torch.stack([centered[k] for k in keys])
        U, S, V = torch.svd(mat)
        print(f"\n  SVD singular values: {S[0]:.1f}, {S[1]:.1f}, {S[2]:.1f}")
        print(f"  Variance explained: {S[0]**2/S.pow(2).sum():.1%},"
              f" {S[1]**2/S.pow(2).sum():.1%},"
              f" {S[2]**2/S.pow(2).sum():.1%}")

        torch.cuda.empty_cache()

    print(f"\n{'='*100}")
    print("DONE")


if __name__ == "__main__":
    main()
