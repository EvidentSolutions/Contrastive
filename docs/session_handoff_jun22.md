# Session Handoff — June 22, 2026

## Paper status

`contrastive/docs/paper_v1.md` — latest commit `b1d4b81` on master.

9 sections: Abstract, Intro, Method, Validation, Lexical disambiguation,
Replicating landmarks, Contrastive axis taxonomy, Ordering mechanism,
Discussion + Related work.

### What was done this session

1. **Per-head analysis** added to §5 (IOI: H1/H14/H16 name-movers, factual:
   H13, successor: H11 discontinuity head)
2. **Validation section (§3)** added: null model (z=64–93 vs random) + causal
   injection (4 cases, z=223–4249) + logit-lens invisibility (0–1/5 overlap)
3. **Smoothness corrected**: unrelated pairs are smoother (0.93±0.01, N=40)
   than minimal pairs (0.88±0.03, N=6). Smoothness is residual stream inertia,
   not content validation. Trimmed to one paragraph in §2.4.
4. **LayerNorm**: bypass LN_f, verified rankings invariant (3 variants, 6
   cases). Cross-layer norms use ||Δh||/||h_c||.
5. **Novelty framing**: explicitly named RepE + Logit Lens connection in intro.
6. **Causal framing**: "subspace contains the causal mechanism" not "probe is
   causal."
7. **Activation patching improved**: multi-layer multi-position table for hot
   dog. L5 transition at dog position is the non-trivial finding (compound
   recognized here). L0 patching removed (trivially = changing input).
8. **Hallucination section (§5.3)**: real entities read factual associations
   (Tesla→alternating), fictional read name fragments (Vog, von). Entity-vs-
   generic confirms. Entropy alone can't distinguish confident recall from
   confident hallucination.
9. **Circular geometry** moved to supplementary (doesn't use the method).
10. **16 issues fixed** from internal review: overclaims softened, section
    numbering fixed, stale limitations updated, duplicate citations removed.
11. **Factual recall trimmed**: removed unfounded real-vs-fictional "clean
    separation" claim. Replaced with France-vs-Japan two-domain contrast.
12. **Cross-layer norm claim**: restored — code does compute ||Δh||/||h_c||
    at every layer, and §5.3 reports it.

### Scripts created this session

- `contrastive/code/per_head_analysis.py` — per-head decomposition for IOI, factual, successor
- `contrastive/code/epistemological.py` — factual vs fictional/hallucinated/temporal
- `contrastive/code/epistemic_verbs.py` — doubt→Berlin, believe→Sydney findings
- `contrastive/code/truth_representation.py` — truth direction (confidence not truth)
- `contrastive/code/layernorm_comparison.py` — raw vs post-LN vs diff-LN comparison
- `contrastive/code/null_and_causal.py` — null model + injection recovery
- `contrastive/code/logit_lens_comparison.py` — logit lens vs contrastive overlap
- `contrastive/code/hallucination_detection.py` — fictional people/places
- `contrastive/code/ioi_head_2x2.py` — 2×2 per-head IOI (MF/MM/FF)
- `contrastive/code/head_logit_lens.py` — single-input head V content (NOT YET RUN — OOM)

---

## Active investigation: attention broadcast hypothesis

### The hypothesis

The tokens we read from contrastive projection at intermediate layers are
the prediction-shaped component of what the residual stream prepares for
attention to broadcast to future positions. Each head's K/V projections
carry different aspects. Different tokens broadcast different properties —
we saw this in IOI (name-movers broadcast names), hot dog (food tokens
broadcast), and ordering (bottom entity broadcast).

### What we tested

**1. V-broadcast decomposition (hot dog, dog position)**

Decomposed the contrastive at the dog position into V→O→W_U path
(what the value projection prepares for broadcast) vs direct residual
reading (what we normally see through W_U).

**Result: V-broadcast does NOT match the W_U reading.**
- Cosine between direct and V-broadcast: 0.03–0.10 at most layers
- At L5 dog position: direct reads "fried", V reads "hawks, pas, mint"
- At L24: direct reads "fried, seasoning, vendor", V reads "hamb, pastry, burger, Sandwich"
- BOTH are in the food domain, but different specific tokens

**2. V-broadcast from dog → was position**

Checked if V-broadcast from dog matches what the was position actually gains.
- Cos(V_from_dog, was_added) = 0.53 at L1 (highest), declining to 0.13 at L8
- The V-broadcast is ONE ingredient of what was receives, mixed with attention
  from other positions and MLP output

**3. IOI V-broadcast (John/Mary)**

At L24 last position:
- Direct contrastive reads "Mary, Mary, himself, wife"
- V-broadcast reads "fathers, husbands, men, dads" — OPPOSITE gender pole!
- At L28: direct reads "Mary", V reads "his, His, himself"

The V prepares gender-context for future positions while the residual
has crystallized onto the specific name.

**Interpretation:** What W_U reads is the accumulated residual state —
sum of all layers' broadcasts + MLP. The V-broadcast at any single layer
is one ingredient, in a different subspace. The domain is preserved (food,
gender) but specific tokens differ.

### What we tested next: 2×2 IOI gender factorial

**{mixed-gender, same-gender-male, same-gender-female} × multiple name pairs**

Key findings:
- Model gets 8/8 same-gender cases correct — gender NOT needed for IOI
- **Different heads activate for different disambiguation types:**
  - Names → H1 (7.5), H14 (8.3), H16 (3.4)
  - Attributes (short/tall man) → H15 "larger, bigger" (2.9), H7 "tall" (2.6)
  - Old/young woman → H21 "youngster, young" (1.4)
  - Red/blue car → almost no signal (norms 0.05-0.09), different mechanism
- H14 direction INVERTS between cases with opposite IO gender (cos=-0.948)
- H1 correlates when same subject name is shared (cos=+0.62)

### Ghost outputs: heads have structural priors

**H16 outputs proper names regardless of input:**
- IOI with names: "Mary, Mary" (norm 3.4) ← expected
- IOI with "Object 1/2": "object, Object" (norm 3.6) ← adapted
- Non-IOI capitals prompt: "Anne, Anne, Bernard, Anthony, Paul" (norm 6.3!) ← names with no IOI structure
- Red/blue car: "Steve, Mike, Joe, Dave, Jane" (norm 0.4) ← ghost names

**H0 outputs "Mary" for red/blue car** (no Mary in prompt, norm 1.0).
Not a universal default — doesn't happen for weather or physics prompts.
The IOI structural pattern partially activates these heads.

### What was NOT yet run (OOM killed it)

`head_logit_lens.py` — logit lens on individual heads for SINGLE inputs
(not contrastive). This would show what each head's V→O actually contains
for one prompt, separating the head's full output from the contrastive
difference. The key question: does H16 output "Mary" for input A AND
something Mary-related for input B, or does one side dominate?

This is the immediate next step. The script is written and ready to run.

---

## Saved for future work (in memory files)

- `epistemological_future_work.md` — truth direction (confidence not truth),
  epistemic verb modulation (doubt→Berlin, believe→Sydney)
- `token_bound_capabilities.md` — ordering: "Ranking: 1." invokes sorting
  but "2nd tallest is" doesn't; capabilities bound to tokens; prior art:
  Li et al. ICLR 2024 (CoT = serial depth)
- `head_structural_priors.md` — this session's head analysis, ghost outputs,
  V-broadcast negative result, content-type specialization

---

## Core open question

The contrastive reads "prediction-shaped content" — but is this content
what the model prepares for attention broadcast, or is it accumulated
residual state that has already been broadcast and processed? The V
decomposition shows it's the latter (accumulated), but the domain is
preserved (food, gender, names). The per-head analysis shows different
heads activate for different content types, suggesting the broadcast
mechanism is content-type-specific. The ghost outputs suggest heads have
structural priors from training that fire even when expected content is
absent.

The next step is single-input head logit lens to see what each head
outputs for one prompt (not contrastive), then compare A's head output
vs B's head output vs A-B contrastive to understand what the contrastive
is actually reading from each head.
