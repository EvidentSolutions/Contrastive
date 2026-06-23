"""
Cluster active MLP neurons by what they write.

For a given prompt and layer:
1. Find all neurons with post-GELU > threshold
2. Compute each neuron's WEIGHTED write direction: act[n] * fc2[:, n]
   (what it actually contributes to the residual stream)
3. Project through W_U to see what each neuron writes
4. Cluster by cosine similarity of fc2 write directions
5. Report: how many clusters, what each cluster writes, how much
   energy each cluster contributes

This answers: how many distinct signals is the MLP writing at this layer?
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")

print(f"Loading {MODEL} on {DEV}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)

NL = model.config.num_hidden_layers
d_model = model.config.hidden_size
d_inter = model.config.intermediate_size
W_U = model.lm_head.weight.detach().float()


def _sl(*layers):
    return sorted(set(min(round(l * NL / 32), NL) for l in layers))


def topk_tok(logits, k=5):
    vals, idxs = torch.topk(logits.float(), k)
    return [tok.decode([int(idxs[j])]).strip()[:14] for j in range(k)]


def get_post_gelu(text, layer_idx):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    fc1_out = {}

    def hook_fn(module, input, output):
        fc1_out['pre_gelu'] = output[0, -1, :].detach().float()

    handle = model.model.layers[layer_idx].mlp.fc1.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(torch.tensor([ids], device=DEV))
    handle.remove()
    gelu = model.model.layers[layer_idx].mlp.activation_fn
    return gelu(fc1_out['pre_gelu'])


def cluster_by_cosine(write_dirs, threshold=0.3):
    """
    Simple greedy clustering: assign each vector to the first cluster
    whose centroid it has cos > threshold with. Otherwise start new cluster.
    """
    clusters = []  # list of (centroid, [indices])

    for i in range(len(write_dirs)):
        v = write_dirs[i]
        assigned = False
        for c_idx, (centroid, members) in enumerate(clusters):
            cos_val = float(F.cosine_similarity(
                v.unsqueeze(0), centroid.unsqueeze(0)))
            if cos_val > threshold:
                # Update centroid as running mean
                n = len(members)
                clusters[c_idx] = (
                    (centroid * n + v) / (n + 1),
                    members + [i]
                )
                assigned = True
                break
        if not assigned:
            clusters.append((v.clone(), [i]))

    return clusters


prompts = [
    ("hot_dog", "The hot dog was"),
    ("caught_cold", "She caught a cold and went to"),
    ("capital", "The capital of France is"),
    ("IOI", "When Mary and John went to the store, John gave a drink to"),
    ("metaphor", "The ice in the bucket was extremely cold. The temperature was"),
    ("some_all", "Some of the students passed the exam, so"),
]

act_threshold = 0.3
cluster_thresholds = [0.3, 0.5]

for name, prompt in prompts:
    print(f"\n{'='*70}")
    print(f"  {name}: \"{prompt[-50:]}\"")
    print(f"{'='*70}")

    for L in [_sl(12)[0], _sl(20)[0], _sl(28)[0]]:
        fc2_w = model.model.layers[L].mlp.fc2.weight.detach().float()
        acts = get_post_gelu(prompt, L)

        # Find active neurons
        active_mask = acts.abs() > act_threshold
        active_idx = active_mask.nonzero(as_tuple=True)[0].tolist()
        n_active = len(active_idx)

        if n_active == 0:
            print(f"\n  L{L}: 0 active neurons (>{act_threshold})")
            continue

        # Get weighted write directions: act[n] * fc2[:, n]
        # and raw write directions: fc2[:, n] (for clustering)
        weighted_writes = []
        raw_writes = []
        activations = []
        for n in active_idx:
            a = float(acts[n])
            w = fc2_w[:, n].to(DEV)
            weighted_writes.append(a * w)
            raw_writes.append(w / (w.norm() + 1e-8))  # normalize for cosine
            activations.append(a)

        # Total MLP output for reference
        total_mlp = torch.stack(weighted_writes).sum(dim=0)
        total_logits = total_mlp @ W_U.T
        total_toks = topk_tok(total_logits, 6)

        print(f"\n  L{L}: {n_active} active neurons (>{act_threshold})")
        print(f"    Total MLP output reads: {total_toks}")

        # Cluster at different thresholds
        for ct in cluster_thresholds:
            raw_stack = torch.stack(raw_writes)
            clusters = cluster_by_cosine(raw_stack, threshold=ct)

            # Sort clusters by total energy (sum of |activation * write_norm|)
            cluster_info = []
            for centroid, members in clusters:
                energy = sum(abs(activations[m]) * float(weighted_writes[m].norm())
                             for m in members)
                # Sum the weighted writes for this cluster
                cluster_vec = torch.stack([weighted_writes[m] for m in members]).sum(dim=0)
                cluster_logits = cluster_vec @ W_U.T
                cluster_toks = topk_tok(cluster_logits, 5)
                cluster_neg = topk_tok(-cluster_logits, 5)
                cluster_info.append((len(members), energy, cluster_toks, cluster_neg))

            cluster_info.sort(key=lambda x: -x[1])

            print(f"\n    Clustering at cos>{ct}: {len(clusters)} clusters")
            for rank, (n_members, energy, toks, neg_toks) in enumerate(cluster_info[:15]):
                print(f"      #{rank+1:>2} ({n_members:>3} neurons, E={energy:>6.1f}): "
                      f"+[{', '.join(toks[:4])}]  -[{', '.join(neg_toks[:3])}]")

            # How many clusters account for 80% of energy?
            total_energy = sum(ci[1] for ci in cluster_info)
            cum = 0
            for i, (_, e, _, _) in enumerate(cluster_info):
                cum += e
                if cum >= total_energy * 0.8:
                    print(f"      → 80% of energy in top {i+1} / {len(clusters)} clusters")
                    break

        torch.cuda.empty_cache()


print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
