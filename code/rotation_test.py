"""
Test rotation hypothesis for ordering representation.

If the model uses rotation to cycle through ranks, the step vectors
(tallest→2nd_tallest, 2nd→3rd, 3rd→shortest) should be related
by a consistent rotation angle.

Usage: .venv/Scripts/python.exe contrastive/code/rotation_test.py
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

    base = (
        "Alice is taller than Bob. Bob is taller than Carol. "
        "Carol is taller than Dan. "
    )
    queries = {
        "tallest": base + "Who is the tallest?\nAnswer:",
        "2nd_tallest": base + "Who is the second tallest?\nAnswer:",
        "3rd_tallest": base + "Who is the third tallest?\nAnswer:",
        "shortest": base + "Who is the shortest?\nAnswer:",
    }

    name_ids = {
        "Alice": tok(" Alice", add_special_tokens=False)["input_ids"][0],
        "Bob": tok(" Bob", add_special_tokens=False)["input_ids"][0],
        "Carol": tok(" Carol", add_special_tokens=False)["input_ids"][0],
        "Dan": tok(" Dan", add_special_tokens=False)["input_ids"][0],
    }

    outs = {}
    for key, text in queries.items():
        ids = tok(text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out = model(
                torch.tensor([ids], device=DEV), output_hidden_states=True
            )
            gen = model.generate(
                torch.tensor([ids], device=DEV),
                max_new_tokens=15,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        outs[key] = (out, ids)
        answer = tok.decode(gen[0][len(ids) :]).strip().split("\n")[0][:40]
        print(f"  {key:>15}: -> {answer}")

    order = ["tallest", "2nd_tallest", "3rd_tallest", "shortest"]
    # Expected answers: Alice, Bob, Carol, Dan

    for L in [24, 28, 32]:
        print(f"\n{'='*100}")
        print(f"Layer {L}")
        print(f"{'='*100}")

        # Name logits at answer position
        print(f"\n  Name logits at answer position:")
        vecs_logit = {}
        for key in order:
            out, ids = outs[key]
            h = out.hidden_states[L][0, -1, :].float()
            logits = h @ W_U.float().T
            nl = {n: float(logits[tid]) for n, tid in name_ids.items()}
            winner = max(nl, key=nl.get)
            print(
                f"    {key:>15}: A={nl['Alice']:>+7.1f} B={nl['Bob']:>+7.1f}"
                f" C={nl['Carol']:>+7.1f} D={nl['Dan']:>+7.1f} -> {winner}"
            )
            vecs_logit[key] = torch.tensor(
                [nl["Alice"], nl["Bob"], nl["Carol"], nl["Dan"]]
            )

        # Full hidden states at answer position
        full_vecs = {}
        for key in order:
            out, ids = outs[key]
            full_vecs[key] = out.hidden_states[L][0, -1, :].float().cpu()

        # Step vectors in full space
        d_12 = full_vecs["2nd_tallest"] - full_vecs["tallest"]
        d_23 = full_vecs["3rd_tallest"] - full_vecs["2nd_tallest"]
        d_34 = full_vecs["shortest"] - full_vecs["3rd_tallest"]

        print(f"\n  Step vectors in full {d_12.shape[0]}-dim space:")
        print(f"    ||tallest->2nd||  = {float(d_12.norm()):.1f}")
        print(f"    ||2nd->3rd||      = {float(d_23.norm()):.1f}")
        print(f"    ||3rd->shortest|| = {float(d_34.norm()):.1f}")

        cos_12_23 = float(
            torch.nn.functional.cosine_similarity(
                d_12.unsqueeze(0), d_23.unsqueeze(0)
            )
        )
        cos_23_34 = float(
            torch.nn.functional.cosine_similarity(
                d_23.unsqueeze(0), d_34.unsqueeze(0)
            )
        )
        cos_12_34 = float(
            torch.nn.functional.cosine_similarity(
                d_12.unsqueeze(0), d_34.unsqueeze(0)
            )
        )

        print(f"    cos(step1, step2) = {cos_12_23:>+.3f}")
        print(f"    cos(step2, step3) = {cos_23_34:>+.3f}")
        print(f"    cos(step1, step3) = {cos_12_34:>+.3f}")
        print(
            f"    If rotation: consecutive cos should be equal,"
            f" step1-step3 cos should be cos^2"
        )

        # SVD of step matrix
        step_mat = torch.stack([d_12, d_23, d_34])
        U, S, V = torch.svd(step_mat)
        print(f"\n  SVD of step vectors:")
        print(f"    Singular values: {S[0]:.1f}, {S[1]:.1f}, {S[2]:.1f}")
        total_var = S.pow(2).sum()
        print(
            f"    Variance: {S[0] ** 2 / total_var:.1%},"
            f" {S[1] ** 2 / total_var:.1%},"
            f" {S[2] ** 2 / total_var:.1%}"
        )

        # Project into top-2 subspace
        V2 = V[:, :2]
        steps_2d = step_mat @ V2
        print(f"\n  Steps projected to top-2 subspace:")
        for i, name in enumerate(
            ["tallest->2nd", "2nd->3rd", "3rd->shortest"]
        ):
            x, y = float(steps_2d[i, 0]), float(steps_2d[i, 1])
            angle = float(
                torch.atan2(torch.tensor(y), torch.tensor(x))
                * 180
                / 3.14159
            )
            norm = float(steps_2d[i].norm())
            print(
                f"      {name:>20}: ({x:>+8.1f}, {y:>+8.1f})"
                f"  norm={norm:.1f}  angle={angle:>+7.1f}deg"
            )

        # Angles between consecutive steps in 2D
        for i in range(2):
            v1 = steps_2d[i]
            v2 = steps_2d[i + 1]
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
            print(f"      angle(step{i+1}, step{i+2}) = {angle:.1f}deg")

        # Also project the QUERY states (not steps) into 2D
        # Center first
        all_h = torch.stack([full_vecs[k] for k in order])
        center = all_h.mean(dim=0)
        centered = all_h - center

        # SVD of centered query states
        U_q, S_q, V_q = torch.svd(centered)
        V2_q = V_q[:, :2]
        proj_2d = centered @ V2_q

        print(f"\n  Query states projected to their own top-2 subspace:")
        for i, key in enumerate(order):
            x, y = float(proj_2d[i, 0]), float(proj_2d[i, 1])
            angle = float(
                torch.atan2(torch.tensor(y), torch.tensor(x))
                * 180
                / 3.14159
            )
            norm = float(proj_2d[i].norm())
            print(
                f"      {key:>15}: ({x:>+8.1f}, {y:>+8.1f})"
                f"  norm={norm:.1f}  angle={angle:>+7.1f}deg"
            )

        # Consecutive angles
        print(f"    Consecutive angular steps:")
        for i in range(3):
            a1 = float(
                torch.atan2(proj_2d[i, 1], proj_2d[i, 0]) * 180 / 3.14159
            )
            a2 = float(
                torch.atan2(proj_2d[i + 1, 1], proj_2d[i + 1, 0])
                * 180
                / 3.14159
            )
            step = a2 - a1
            if step > 180:
                step -= 360
            if step < -180:
                step += 360
            print(
                f"      {order[i]:>15} -> {order[i+1]:<15}:"
                f" {step:>+7.1f}deg"
            )

        # SVD variance of centered query states
        print(f"    SVD of centered query states:")
        total_var_q = S_q.pow(2).sum()
        for i in range(min(4, len(S_q))):
            pct = S_q[i] ** 2 / total_var_q
            print(f"      S{i+1} = {S_q[i]:.1f} ({pct:.1%})")

    keys = list(outs.keys())
    for key in keys:
        del outs[key]
    torch.cuda.empty_cache()

    print(f"\n{'='*100}")
    print("DONE")


if __name__ == "__main__":
    main()
