"""
Revised: group active neurons by what TOKEN they write to,
not by fc2 direction similarity.

Two neurons with different fc2 vectors can still write to the same
token if their fc2 columns both project to the same top-1 W_U token.

Also: energy distribution — is it heavy-tailed? Do a few neurons
dominate the output?
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
        fc1_out['v'] = output[0, -1, :].detach().float()
    handle = model.model.layers[layer_idx].mlp.fc1.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(torch.tensor([ids], device=DEV))
    handle.remove()
    gelu = model.model.layers[layer_idx].mlp.activation_fn
    return gelu(fc1_out['v'])


prompts = [
    ("hot_dog", "The hot dog was"),
    ("caught_cold", "She caught a cold and went to"),
    ("capital", "The capital of France is"),
    ("IOI", "When Mary and John went to the store, John gave a drink to"),
]

act_threshold = 0.3

for name, prompt in prompts:
    print(f"\n{'='*70}")
    print(f"  {name}: \"{prompt}\"")
    print(f"{'='*70}")

    for L in [_sl(12)[0], _sl(20)[0], _sl(28)[0]]:
        fc2_w = model.model.layers[L].mlp.fc2.weight.detach().float()
        acts = get_post_gelu(prompt, L)

        active_mask = acts.abs() > act_threshold
        active_idx = active_mask.nonzero(as_tuple=True)[0].tolist()
        n_active = len(active_idx)

        if n_active == 0:
            continue

        # ── Energy distribution ──
        # For each active neuron: contribution = act[n] * fc2[:, n]
        # Energy = ||contribution|| = |act[n]| * ||fc2[:, n]||
        energies = []
        contributions = []
        for n in active_idx:
            a = float(acts[n])
            w = fc2_w[:, n].to(DEV)
            contrib = a * w
            e = float(contrib.norm())
            energies.append((n, a, e, contrib))
            contributions.append(contrib)

        # Sort by energy
        energies.sort(key=lambda x: -x[2])
        total_energy = sum(e for _, _, e, _ in energies)

        # How many neurons for 50%, 80%, 90%?
        cum = 0
        thresholds_hit = {}
        for i, (n, a, e, _) in enumerate(energies):
            cum += e
            for t in [0.5, 0.8, 0.9]:
                if t not in thresholds_hit and cum >= total_energy * t:
                    thresholds_hit[t] = i + 1

        print(f"\n  L{L}: {n_active} active neurons")
        print(f"    Energy concentration: "
              f"50% in top {thresholds_hit.get(0.5, '?')}, "
              f"80% in top {thresholds_hit.get(0.8, '?')}, "
              f"90% in top {thresholds_hit.get(0.9, '?')}")

        # ── Top 15 neurons by energy — what token do they write? ──
        print(f"    Top neurons by energy (weighted write through W_U):")
        for rank, (n, a, e, contrib) in enumerate(energies[:15]):
            logits = contrib @ W_U.T
            top1 = topk_tok(logits, 3)
            bot1 = topk_tok(-logits, 2)
            pct = e / total_energy * 100
            print(f"      #{rank+1:>2} N{n:>5} act={a:>+5.2f} E={e:>5.1f} ({pct:>4.1f}%)  "
                  f"+[{', '.join(top1)}] -[{', '.join(bot1)}]")

        # ── Group by top-1 token ──
        token_groups = defaultdict(list)  # top1_token_id -> [(neuron, activation, energy)]
        for n, a, e, contrib in energies:
            logits = contrib @ W_U.T
            if a > 0:
                top1_id = int(torch.argmax(logits))
            else:
                top1_id = int(torch.argmin(logits))
            token_groups[top1_id].append((n, a, e))

        # Sort groups by total energy
        group_list = []
        for tid, members in token_groups.items():
            group_energy = sum(e for _, _, e in members)
            token_name = tok.decode([tid]).strip()[:14]
            group_list.append((token_name, tid, len(members), group_energy, members))
        group_list.sort(key=lambda x: -x[3])

        print(f"\n    Token groups (neurons grouped by top-1 W_U output):")
        print(f"    {len(group_list)} unique tokens written by {n_active} neurons")
        for rank, (tname, tid, n_members, g_energy, members) in enumerate(group_list[:20]):
            pct = g_energy / total_energy * 100
            print(f"      '{tname:>12}': {n_members:>3} neurons, "
                  f"E={g_energy:>5.1f} ({pct:>4.1f}%)")

        # ── How many token groups for 80% energy? ──
        cum = 0
        for i, (_, _, _, ge, _) in enumerate(group_list):
            cum += ge
            if cum >= total_energy * 0.8:
                print(f"    → 80% of energy in top {i+1} / {len(group_list)} token groups")
                break

        # ── What does the TOTAL sum read as? ──
        total_vec = torch.stack([c for c in contributions]).sum(dim=0)
        total_logits = total_vec @ W_U.T
        print(f"    Total MLP output: +{topk_tok(total_logits, 5)} -{topk_tok(-total_logits, 4)}")

        torch.cuda.empty_cache()


print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
