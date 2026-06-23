"""
Sweep all attention heads: which ones do double duty?

For each head at each layer, collect V[noun]@O_h across many
"The [adj] [noun] was" prompts. Decompose into:
  - Mean (constant component) → structural role
  - Top PCs of centered output → content features

A "double duty" head has:
  - Large mean (structural signal) that reads as structural tokens
  - Large variance (content signal) where PCs read as content tokens
  - Mean and top PCs are roughly orthogonal

Usage: .venv/Scripts/python.exe contrastive/code/head_dual_roles.py
"""
import sys
import torch
import torch.nn.functional as F

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
    for p in model.parameters(): p.requires_grad_(False)

    NL = model.config.num_hidden_layers
    NH = model.config.num_attention_heads
    HD = model.config.hidden_size // NH
    HIDDEN = model.config.hidden_size

    if hasattr(model, "lm_head"):
        W_U = model.lm_head.weight.detach().float().cpu()
    else:
        W_U = model.embed_out.weight.detach().float().cpu()

    DENSE_ATTR = "dense" if hasattr(model.model.layers[0].self_attn, "dense") else "o_proj"

    def tk(logits, k=4):
        v, i = torch.topk(logits.float(), k)
        return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))

    adjectives = ["hot", "cold", "old", "young", "angry", "happy",
                  "big", "small", "red", "blue", "new", "broken",
                  "fresh", "raw", "fried", "grilled"]
    nouns = ["dog", "cat", "rod", "car", "chicken", "fish",
             "burger", "pizza", "water", "chocolate"]

    noun_pos = 2  # "The [adj] [noun] was" → noun at pos 2

    # Pre-compute all hidden states at each layer's input
    print(f"Collecting hidden states for {len(adjectives)}×{len(nouns)} prompts...")
    all_hidden = {}  # (adj, noun) → {L: hidden_state_at_noun_pos}
    for adj in adjectives:
        for noun in nouns:
            text = f"The {adj} {noun} was"
            ids = tok(text, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                out = model(torch.tensor([ids], device=DEV),
                            output_hidden_states=True)
            states = {}
            for L in range(NL):
                states[L] = out.hidden_states[L][0, noun_pos, :].float().cpu()
            all_hidden[(adj, noun)] = states
            del out
        torch.cuda.empty_cache()

    print(f"  {len(all_hidden)} prompts collected")

    # Sweep all heads at all layers
    print(f"\n{'='*120}")
    print(f"HEAD SWEEP: constant (structural) vs variable (content) for each head")
    print(f"{'='*120}")

    # Collect summary for ranking
    head_summaries = []

    for L in range(NL):
        layer_mod = model.model.layers[L]
        dense = getattr(layer_mod.self_attn, DENSE_ATTR)
        O_w = dense.weight.float().cpu()
        ln = layer_mod.input_layernorm if hasattr(layer_mod, 'input_layernorm') else None

        # Get W_V for this layer
        if hasattr(layer_mod.self_attn, 'qkv_proj'):
            W_qkv = layer_mod.self_attn.qkv_proj.weight.float().cpu()
            b_qkv = layer_mod.self_attn.qkv_proj.bias
            b_qkv = b_qkv.float().cpu() if b_qkv is not None else None
            W_V = W_qkv[2*HIDDEN:3*HIDDEN, :]
            b_V = b_qkv[2*HIDDEN:3*HIDDEN] if b_qkv is not None else None
        elif hasattr(layer_mod.self_attn, 'v_proj'):
            W_V = layer_mod.self_attn.v_proj.weight.float().cpu()
            b_V = getattr(layer_mod.self_attn.v_proj, 'bias', None)
            if b_V is not None: b_V = b_V.float().cpu()
        else:
            continue

        for h in range(NH):
            O_h = O_w[:, h*HD:(h+1)*HD]
            W_V_h = W_V[h*HD:(h+1)*HD, :]

            # Collect V[noun]@O_h for all prompts
            outputs = []
            for adj in adjectives:
                for noun in nouns:
                    h_input = all_hidden[(adj, noun)][L]
                    if ln is not None:
                        with torch.no_grad():
                            normed = ln(h_input.half().unsqueeze(0).to(DEV)).float().cpu().squeeze()
                    else:
                        normed = h_input
                    v_h = normed @ W_V_h.T
                    if b_V is not None:
                        v_h = v_h + b_V[h*HD:(h+1)*HD]
                    contrib = v_h @ O_h.T
                    outputs.append(contrib)

            all_out = torch.stack(outputs)  # (N, HIDDEN)
            mean_out = all_out.mean(dim=0)
            centered = all_out - mean_out

            mean_norm = float(mean_out.norm())
            var_norm = float(centered.pow(2).sum(dim=1).mean().sqrt())
            total_norm = float(all_out.pow(2).sum(dim=1).mean().sqrt())

            if total_norm < 0.5:
                continue  # skip tiny heads

            mean_frac = mean_norm**2 / (total_norm**2 + 1e-10)
            var_frac = 1 - mean_frac

            # SVD of centered to get top PCs
            if centered.shape[0] > 3:
                U, S, V = torch.svd(centered)
                top_var = float(S[0]**2 / (S.pow(2).sum() + 1e-10))
                pc1 = V[:, 0]
                pc1_wu = pc1 @ W_U.T
                # cos between mean and PC1
                if mean_norm > 0.1 and float(S[0]) > 0.1:
                    cos_mean_pc1 = float(F.cosine_similarity(
                        mean_out.unsqueeze(0), pc1.unsqueeze(0)))
                else:
                    cos_mean_pc1 = 0.0
            else:
                top_var = 0; pc1_wu = torch.zeros(W_U.shape[0])
                cos_mean_pc1 = 0

            mean_wu = mean_out @ W_U.T

            head_summaries.append({
                "L": L, "H": h,
                "mean_norm": mean_norm,
                "var_norm": var_norm,
                "total_norm": total_norm,
                "mean_frac": mean_frac,
                "var_frac": var_frac,
                "mean_wu": tk(mean_wu),
                "mean_wu_neg": tk(-mean_wu),
                "pc1_wu": tk(pc1_wu),
                "pc1_wu_neg": tk(-pc1_wu),
                "pc1_var": top_var,
                "cos_mean_pc1": cos_mean_pc1,
            })

    # ================================================================
    # Print top "double duty" heads: large mean AND large variance,
    # with mean and PC1 roughly orthogonal
    # ================================================================
    print(f"\n{'='*120}")
    print(f"TOP DUAL-ROLE HEADS (sorted by min(mean_norm, var_norm), |cos|<0.3)")
    print(f"{'='*120}")

    # Score: a head is "dual" if both mean and variance are large
    # and they're roughly orthogonal
    for entry in head_summaries:
        entry["dual_score"] = min(entry["mean_norm"], entry["var_norm"])

    dual = [e for e in head_summaries
            if abs(e["cos_mean_pc1"]) < 0.3 and e["dual_score"] > 2.0]
    dual.sort(key=lambda x: -x["dual_score"])

    print(f"\n  {'L.H':>5} {'||mean||':>8} {'||var||':>7} {'mean%':>6} "
          f"{'cos':>5} {'PC1 var%':>8}  "
          f"{'mean reads as':>35}  {'PC1 reads as':>35}")

    for e in dual[:30]:
        print(f"  L{e['L']:>1}.H{e['H']:<2} {e['mean_norm']:>8.1f} "
              f"{e['var_norm']:>7.1f} {e['mean_frac']:>5.0%} "
              f"{e['cos_mean_pc1']:>+5.2f} {e['pc1_var']:>7.0%}  "
              f"{e['mean_wu']:>35}  {e['pc1_wu']:>35}")

    # ================================================================
    # Also show heads with very high mean fraction (structural only)
    # and very high var fraction (content only)
    # ================================================================
    print(f"\n{'='*120}")
    print(f"STRUCTURAL-DOMINANT HEADS (mean > 70% of total energy, ||mean|| > 3)")
    print(f"{'='*120}")

    structural = [e for e in head_summaries
                  if e["mean_frac"] > 0.7 and e["mean_norm"] > 3]
    structural.sort(key=lambda x: -x["mean_norm"])

    for e in structural[:15]:
        print(f"  L{e['L']:>1}.H{e['H']:<2} ||mean||={e['mean_norm']:>5.1f} "
              f"mean%={e['mean_frac']:>4.0%}  "
              f"reads: [{e['mean_wu']}]")

    print(f"\n{'='*120}")
    print(f"CONTENT-DOMINANT HEADS (var > 85% of energy, ||var|| > 5)")
    print(f"{'='*120}")

    content = [e for e in head_summaries
               if e["var_frac"] > 0.85 and e["var_norm"] > 5]
    content.sort(key=lambda x: -x["var_norm"])

    for e in content[:15]:
        print(f"  L{e['L']:>1}.H{e['H']:<2} ||var||={e['var_norm']:>5.1f} "
              f"var%={e['var_frac']:>4.0%}  "
              f"PC1({e['pc1_var']:.0%}): +[{e['pc1_wu']}]  -[{e['pc1_wu_neg']}]")

    torch.cuda.empty_cache()
    print(f"\n{'='*120}")
    print("DONE")


if __name__ == "__main__":
    main()
