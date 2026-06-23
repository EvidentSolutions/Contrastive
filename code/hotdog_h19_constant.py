"""
What is the constant component of H19's output at dog position?

H19@L5 writes norm ~6.8 to the residual stream. The contrastive
(hot-cold) reads as "eaten, cooked" but individual writes read as
junk. What is the large shared component?

1. H19 output for many "The [adj] [noun] was" — what's constant?
2. Mean output across all variants (the "background")
3. Variance structure: is it noun-dependent? adj-dependent? position-dependent?
4. Same analysis for H17 (the food head) for comparison

Usage: .venv/Scripts/python.exe contrastive/code/hotdog_h19_constant.py
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
    for p in model.parameters():
        p.requires_grad_(False)

    NL = model.config.num_hidden_layers
    NH = model.config.num_attention_heads
    HD = model.config.hidden_size // NH
    HIDDEN = model.config.hidden_size

    if hasattr(model, "lm_head"):
        W_U = model.lm_head.weight.detach().float().cpu()
    else:
        W_U = model.embed_out.weight.detach().float().cpu()

    DENSE_ATTR = "dense" if hasattr(model.model.layers[0].self_attn, "dense") else "o_proj"

    def tk(logits, k=6):
        v, i = torch.topk(logits.float(), k)
        return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))

    def _sl(*layers):
        return sorted(set(min(round(l * NL / 32), NL) for l in layers))

    L = 5
    layer_mod = model.model.layers[L]
    dense = getattr(layer_mod.self_attn, DENSE_ATTR)
    O_w = dense.weight.float().cpu()
    ln = layer_mod.input_layernorm if hasattr(layer_mod, 'input_layernorm') else None

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

    def get_head_output(text, pos, head):
        """Get head h's V[pos]@O_h contribution for a single input."""
        ids = tok(text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out = model(torch.tensor([ids], device=DEV),
                        output_hidden_states=True)
        h_input = out.hidden_states[L][0, pos, :].float()
        del out; torch.cuda.empty_cache()

        if ln is not None:
            with torch.no_grad():
                normed = ln(h_input.half().unsqueeze(0).to(DEV)).float().cpu().squeeze()
        else:
            normed = h_input.cpu()

        V_full = normed @ W_V.T + (b_V if b_V is not None else 0)
        v_h = V_full[head*HD:(head+1)*HD]
        O_h = O_w[:, head*HD:(head+1)*HD]
        return (v_h @ O_h.T).cpu()  # (HIDDEN,)

    # ================================================================
    # 1. Collect H19 and H17 outputs for many variants at noun pos
    # ================================================================
    adjectives = ["hot", "cold", "old", "young", "angry", "happy",
                  "big", "small", "red", "blue", "new", "broken",
                  "fresh", "raw", "fried", "grilled"]
    nouns = ["dog", "cat", "rod", "car", "chicken", "fish",
             "burger", "pizza", "water", "chocolate"]

    for HEAD in [19, 17]:
        print(f"\n{'='*100}")
        print(f"HEAD {HEAD} @ L{L} — output at noun position for many variants")
        print(f"{'='*100}")

        outputs = {}
        for adj in adjectives:
            for noun in nouns:
                text = f"The {adj} {noun} was"
                ids = tok(text, add_special_tokens=False)["input_ids"]
                tokens = [tok.decode([t]) for t in ids]
                noun_pos = next((i for i, t in enumerate(tokens)
                                 if noun.lower() in t.lower().strip()), None)
                if noun_pos is None: continue
                h_out = get_head_output(text, noun_pos, HEAD)
                outputs[f"{adj}_{noun}"] = h_out

        all_vecs = torch.stack(list(outputs.values()))
        mean_vec = all_vecs.mean(dim=0)
        centered = all_vecs - mean_vec

        # ================================================================
        # What does the mean (constant component) read as?
        # ================================================================
        ld_mean = mean_vec @ W_U.T
        print(f"\n  Mean output across all {len(outputs)} variants:")
        print(f"    ||mean|| = {float(mean_vec.norm()):.1f}")
        print(f"    W_U reads: [{tk(ld_mean)}]")
        print(f"    W_U -pole: [{tk(-ld_mean)}]")

        # ================================================================
        # How much variance is in the mean vs the deviation?
        # ================================================================
        mean_norm_sq = float(mean_vec.norm()**2)
        var_norms = float(centered.pow(2).sum() / len(outputs))
        total_norms = float(all_vecs.pow(2).sum() / len(outputs))
        print(f"\n  Variance decomposition:")
        print(f"    ||mean||² = {mean_norm_sq:.1f}  ({mean_norm_sq/total_norms:.1%} of total)")
        print(f"    mean var  = {var_norms:.1f}  ({var_norms/total_norms:.1%} of total)")
        print(f"    total     = {total_norms:.1f}")

        # ================================================================
        # SVD of centered outputs — what are the principal variations?
        # ================================================================
        U, S, V = torch.svd(centered)
        total_var = S.pow(2).sum()
        print(f"\n  SVD of centered head outputs (what varies across variants):")
        for i in range(min(6, len(S))):
            pct = S[i]**2 / total_var
            pc = V[:, i]
            ld = pc @ W_U.T
            print(f"    PC{i+1} (var={pct:.1%}): +[{tk(ld, 4)}]  -[{tk(-ld, 4)}]")

        # ================================================================
        # Is it noun-dependent or adj-dependent?
        # ================================================================
        print(f"\n  Mean output by noun (deviation from grand mean):")
        for noun in nouns:
            noun_vecs = [outputs[f"{adj}_{noun}"] for adj in adjectives
                         if f"{adj}_{noun}" in outputs]
            if not noun_vecs: continue
            noun_mean = torch.stack(noun_vecs).mean(dim=0)
            dev = noun_mean - mean_vec
            ld = dev @ W_U.T
            cos_with_mean = float(F.cosine_similarity(
                noun_mean.unsqueeze(0), mean_vec.unsqueeze(0)))
            print(f"    {noun:>12}: ||dev||={float(dev.norm()):>5.1f}  "
                  f"cos(with mean)={cos_with_mean:.4f}  "
                  f"dev reads: [{tk(ld, 4)}]")

        print(f"\n  Mean output by adjective (deviation from grand mean):")
        for adj in adjectives:
            adj_vecs = [outputs[f"{adj}_{noun}"] for noun in nouns
                        if f"{adj}_{noun}" in outputs]
            if not adj_vecs: continue
            adj_mean = torch.stack(adj_vecs).mean(dim=0)
            dev = adj_mean - mean_vec
            ld = dev @ W_U.T
            cos_with_mean = float(F.cosine_similarity(
                adj_mean.unsqueeze(0), mean_vec.unsqueeze(0)))
            print(f"    {adj:>12}: ||dev||={float(dev.norm()):>5.1f}  "
                  f"cos(with mean)={cos_with_mean:.4f}  "
                  f"dev reads: [{tk(ld, 4)}]")

        # ================================================================
        # Show a few individual outputs to see the noise
        # ================================================================
        print(f"\n  Individual outputs (first 8):")
        for key in list(outputs.keys())[:8]:
            h = outputs[key]
            ld = h @ W_U.T
            print(f"    {key:>20} (||={float(h.norm()):>5.1f}): [{tk(ld, 5)}]")

    # ================================================================
    # Also check: what does H19 output at DIFFERENT positions?
    # ================================================================
    print(f"\n{'='*100}")
    print(f"H19 @ L5 — output at DIFFERENT positions in 'The hot dog was'")
    print(f"{'='*100}")

    text = "The hot dog was"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    tokens = [tok.decode([t]) for t in ids]

    for pos in range(len(ids)):
        h_out = get_head_output(text, pos, 19)
        ld = h_out @ W_U.T
        print(f"  pos {pos} \"{tokens[pos]}\": ||={float(h_out.norm()):>5.1f}  [{tk(ld)}]")

    # And for a completely different prompt
    print(f"\n  H19 @ L5 for different prompts:")
    for text in ["The capital of France is",
                 "She walked to the store and",
                 "2 + 3 =",
                 "The quick brown fox"]:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        last_pos = len(ids) - 1
        h_out = get_head_output(text, last_pos, 19)
        ld = h_out @ W_U.T
        cos_with_dog_mean = float(F.cosine_similarity(
            h_out.unsqueeze(0), mean_vec.unsqueeze(0)))
        print(f"  \"{text}\" pos {last_pos}: ||={float(h_out.norm()):>5.1f}  "
              f"cos(mean)={cos_with_dog_mean:.3f}  [{tk(ld, 4)}]")

    # ================================================================
    # The V bias contributes a constant regardless of input
    # ================================================================
    print(f"\n{'='*100}")
    print(f"V BIAS CONTRIBUTION — what does the bias alone produce?")
    print(f"{'='*100}")

    if b_V is not None:
        for HEAD in [19, 17]:
            v_bias = b_V[HEAD*HD:(HEAD+1)*HD]
            O_h = O_w[:, HEAD*HD:(HEAD+1)*HD]
            bias_contrib = v_bias @ O_h.T
            ld = bias_contrib @ W_U.T
            print(f"  H{HEAD} bias-only: ||={float(bias_contrib.norm()):.1f}  [{tk(ld)}]")
    else:
        print("  No V bias")

    torch.cuda.empty_cache()
    print(f"\n{'='*100}")
    print("DONE")


if __name__ == "__main__":
    main()
