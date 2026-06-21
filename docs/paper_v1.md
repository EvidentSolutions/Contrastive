# Contrastive Projection: A Training-Free Probe for Reading Transformer Internals

**Olli Tuomi**, [Affiliation]

---

## Abstract

We describe a training-free method for reading how a transformer's internal
representation differs between two inputs. Subtracting hidden states at each
layer and projecting through the unembedding matrix W_U yields a contrastive
trajectory — a layer-by-layer readout of what separates the two residual
streams in token space. The method requires one matrix multiply per
layer-position pair, no probes to train, and no learned parameters.

We demonstrate the method on Phi-2 (2.7B) as an exploratory tool that
pinpoints where and when the model computes specific distinctions:

**Lexical disambiguation.** "The hot dog was" vs "The cold dog was" — the
contrastive projection traces compound-noun recognition to a specific position
and layer. Activation patching confirms the traced circuit.

**Replication of landmark findings.** The method recovers the key observations
from IOI circuit analysis (IO name crystallizes at L24; per-head decomposition
identifies name-mover heads), factual recall tracing, and successor head
identification. It locates the same phenomena that previously required
circuit-level reverse engineering, serving as a rapid scout for where to
apply heavier mechanistic tools.

The projection reads the prediction-shaped component of the representation —
content aligned with W_U's token rows. We bypass the final LayerNorm; token
rankings are empirically invariant to this choice. Content validation comes
from causal injection (z = 223–4249 across four cases) and logit-lens
invisibility (contrastive top-5 absent from both constituents' top-20 at
mid-layers). The method is silent on non-prediction-shaped computation. We
release code and data.

---

## 1. Introduction

A transformer processing "The hot dog was" predicts continuations about food —
"too hot," "cooked," "served." The same model processing "The cold dog was"
predicts continuations about an animal — "shivering," "panting," "given a
blanket." The word "hot" changes the meaning of "dog" from animal to food item.
We ask: where in the network does this disambiguation happen, and which
component does it?

The logit lens (nostalgebraist 2020) and tuned lens (Belrose et al. 2023)
project individual hidden states through W_U. At intermediate layers they decode
to shared function words — the same for both "hot dog" and "cold dog." The
contrastive projection subtracts one state from the other before projecting,
revealing content that separates the two inputs. We verify this directly: at
layers 8–24 for the hot dog case, the logit lens on both constituents reads
"not, no, made, a, more" (shared function words), while the contrastive
projection reads "fried, crispy, delicious, flavor" — food vocabulary that
appears in neither constituent's top-20 (0–1/5 overlap through L24). Across
five cases, contrastive top-5 tokens overlap with the constituent's logit-lens
top-20 at 0–1/5 for mid-layers, rising to 1–3/5 only at L28+ where the
prediction has crystallized.

The probe is mechanical: given two inputs, subtract their hidden states at each
layer and project the difference through W_U. The most positive tokens
identify content associated with input *c*; the most negative identify content
associated with input *k*. Together, the two poles at each layer form the
trajectory. The arithmetic is identical to taking a RepE/ActAdd steering vector
(Zou et al. 2023; Turner et al. 2023) and reading it through a logit lens
(nostalgebraist 2020). The novelty is not the math but the systematization:
per-position tracing, per-head decomposition, and the use of this operation as
a diagnostic trajectory across all layers rather than a one-shot steering
direction. We make two claims:

1. **Descriptive.** The difference between two matched inputs, projected through
   W_U at each layer and position, reads coherent token-space content that
   changes across layers. The content is set by the chosen pair; the
   model-relevant observations are the layer-wise dynamics and the per-position
   information flow.
2. **Exploratory.** By reading the contrast at every position, we can locate
   which position first computes a meaning distinction, which component
   (attention or MLP) contributes it, and how information flows between
   positions. This provides the "where and when" — pinpointing exactly where
   to apply heavier tools like activation patching or circuit analysis to
   establish the "how."

Primary results use Phi-2 (Microsoft, 2.7B parameters, 32 layers).
Cross-model replication uses Pythia-410M, Pythia-1.4B, and Phi-4 (14B).

---

## 2. Method

### 2.1 Contrastive projection

Given two inputs *c* and *k*:

1. Run both and extract hidden states at every layer at the read position:
   h_c[L] and h_k[L] for L = 0, …, N.
2. Compute Δh[L] = h_c[L] − h_k[L].
3. Project: Δlogits[L] = Δh[L] · W_U^T.
4. Read the top-K most positive tokens (associated with input *c*) and the
   top-K most negative tokens (associated with input *k*). The two poles
   together describe the contrast in token space.

No parameters are fit. The choices are the input pair, the read position, and K.

**LayerNorm.** The model's forward pass applies a final LayerNorm (LN_f) before
W_U. We project raw hidden states, bypassing LN_f. This is deliberate: LN_f
was trained to normalize hidden states at the final layer, and applying it to
intermediate layers imposes statistics (mean, variance) from a distribution it
was not fit to. The raw projection gives W_U a vector at the wrong scale; the
direction is approximately preserved, as verified empirically below.

We verify empirically that bypassing LN_f does not affect token rankings. We
compare three variants across six poster cases (IOI, hot dog, factual recall,
successor, truth, negation) at layers 4–32:

- **Raw**: (h_c − h_k) · W_U^T (our default)
- **Post-norm**: (LN_f(h_c) − LN_f(h_k)) · W_U^T (normalize each, then subtract)
- **Diff-norm**: LN_f(h_c − h_k) · W_U^T (normalize the difference)

All three produce the same top-5 tokens. Top-10 overlap between Raw and
Post-norm is 8–10/10 at L24+ and 6–8/10 at mid-layers. The cosine between
Raw and Post-norm logit vectors exceeds 0.98 at L28+ and 0.84 at the minimum
(L20–24).

The variants diverge only in *magnitude*: residual-stream norms grow across
layers (from 4 at L1 to 175 at L31), so raw Δlogits norms are not comparable
across layers. Where we report norms across layers (e.g., §5.3), we use the
relative norm ||Δh|| / ||h_c|| to remove this scale artifact.

### 2.2 Per-position reading

Under causal attention, the hidden state at position *p* depends only on tokens
at positions ≤ p. For two inputs that share a prefix and differ at position *d*,
the hidden states are identical at all positions < d. At position d and beyond,
the contrast is nonzero and carries information about the differing content.

By reading the contrast at every position, we trace information flow: at which
position does a meaning distinction first appear? Does it appear at the
differing token itself, or at a later position that attends to it?

### 2.3 Attention and MLP decomposition

Each transformer layer adds two components to the residual stream: attention
output and MLP output. We capture both via forward hooks and compute their
contrastive projections separately. This identifies which component writes a
given content distinction at each layer — without causal intervention.

We note that a large contrastive norm in either component indicates that the
component's output differs between the two inputs, not that the component causes
the distinction. Causal verification (activation patching) is required to
establish causality.

### 2.4 What the projection reads

The contrastive projection reads the prediction-shaped component of the
representation difference — the part aligned with W_U's token rows. What W_U
reports as token labels at intermediate layers are not predictions but
continuation preparations: tokens the model might produce at the current or
future positions, modulated by context.

W_U contributes the interpretable token labels. Trajectory smoothness
(consecutive-layer cosine ~0.9) is a general property of the residual stream,
not specific to meaningful contrasts: unrelated pairs ("The hot dog was" vs
"Quantum mechanics is") are equally smooth (0.93 ± 0.01, N=40) as minimal
pairs (0.88 ± 0.03, N=6), because any two residual streams diverge gradually.
Random directions give cosine ≈ 0.00 (z = 64–93), confirming Δh is structured
rather than noise, but smoothness alone does not validate content.

The method is silent on non-prediction-shaped computation — structure in the
residual stream that is not aligned with W_U's token rows.

---

## 3. Validation

### 3.1 The subspace identified by the probe contains the causal mechanism

The probe is exploratory — it reads a difference, not a cause. But we can test
whether the subspace it identifies contains causally relevant information by
injecting Δh[L] from the context into the control's residual stream at layer L
and measuring how much of the prediction gap it recovers: (P_injected −
P_control) / (P_context − P_control) × 100%, compared against 20 random
directions of matched norm.

| Case | L4 | L12 | L20 | L24 | L28 | L31 | Peak z |
|------|-----|------|------|------|------|------|--------|
| hot dog → delicious | 0% | 5% | 73% | 80% | 133% | 150% | 4249 |
| IOI → Mary | 0% | 0% | 0% | 5% | 75% | 113% | 525 |
| Eiffel → Paris | 0% | 0% | 2% | 55% | 73% | 66% | 1778 |
| Successor → Tuesday | 0% | 0% | 0% | 11% | 87% | 98% | 223 |

All four cases show zero recovery at early layers, onset at the layer where
the contrastive projection first reads coherent content, and near-complete
or over-complete recovery by L28–31. Random directions of the same norm give
zero mean recovery at every layer (z-scores measure how far the real direction
exceeds this null).

Recovery exceeding 100% is expected: Δh contains by construction everything
that makes the context predict differently from the control, so injecting it
can overshoot. The operative validation is the comparison to random — the
*direction* matters, not merely the norm.

**Onset matches trajectory content.** Hot dog recovery begins at L12 (5%),
rising sharply at L20 (73%) — the layers where food vocabulary appears in the
contrastive projection at the prediction position. IOI recovers from L24
(where "Mary" crystallizes). Factual recall recovers from L20 (where "France"
first appears). The causal onset tracks the layer at which the contrastive
projection first reads target-relevant content at the prediction position.

---

## 4. Lexical disambiguation

### 4.1 Compound noun: hot dog

**Prompts:**
- "The hot dog was" (food item)
- "The cold dog was" (cold animal)

**Predictions:** hot dog → "too" (0.088), "more" (0.080), "cooked" (0.063);
cold dog → "sh[ivering]" (0.447), "shaking" (0.036).

**Per-position trace (4 tokens: The, hot/cold, dog, was):**

At position 2 ("dog") — same token in both prompts, L0 contrast is zero:
- L1: "hot, molten, fiery" — "dog" receives temperature info from "hot"
- L5: "fried" first appears — compound meaning recognized
- L24: "fried, seasoning, Flav, Serv, vendor" — full food vocabulary
- L32: "vendor, vendors, stand, topping" — hot dog stand

At position 3 ("was") — reads from "dog":
- L6: "fried, delicious, tasty, breakfast" — one layer after "dog" recognizes the compound
- L28: "tasty, charred, crispy, delicious, spicy" (hot) vs "grooming, shudder, shaking" (cold)

**Attention weights at L5, position "was":**
Head 19 attends to "dog" position with weight 0.911 in the hot-dog context vs
0.735 in cold-dog. Head 29 attends 0.345 vs 0.138. These heads read the
compound-noun information from "dog."

**Activation patching at multiple layers and positions:**

| Position patched | Layer | P(food) | Top-1 | Effect |
|-----------------|-------|---------|-------|--------|
| (baseline hot dog) | — | 0.142 | "too" | — |
| dog (pos 2) | L0–L4 | 0.000 | "pant" | Food eliminated |
| dog (pos 2) | L5 | 0.094 | "more" | Food partially survives |
| dog (pos 2) | L12+ | 0.12–0.14 | "too" | Near baseline |
| hot (pos 1) | L0–L28 | 0.11–0.14 | "too" | Minimal effect at all layers |
| was (pos 3) | L0–L5 | 0.13–0.14 | "more" | Food preserved |
| was (pos 3) | L12 | 0.058 | "placed" | Food declining |
| was (pos 3) | L20+ | 0.005 | "sh" | Food eliminated |

The patching traces the two-hop chain precisely. (1) Patching "dog" at L0–L4
eliminates the food signal entirely — the compound meaning has not yet been
computed. At L5 (where "fried" first appears in the contrastive projection),
food partially survives the patch — the computation is happening at this
layer. By L12+ the patch has minimal effect. (2) Patching "hot" has no effect
at any layer — its information has already been copied to "dog" via L0
attention. (3) Patching "was" has no effect at L0–L5 (the food information
hasn't arrived yet) but eliminates food from L12 onward — confirming that
attention at L6+ copies the compound meaning from "dog" to "was."

**Mechanism:** Attention at L0 copies "hot" to the "dog" position. The compound
is recognized at the "dog" position at L5 ("fried" first appears). Attention at
L6+ reads this from "dog" to "was." The contrastive projection identified each
stage; activation patching at three positions and multiple layers confirms the
information flow.

### 4.2 Other disambiguation cases

The same method traces disambiguation in verb-object pairs and noun ambiguity:

**"He caught a cold and" vs "He caught a fish and"** (verb meaning changes):
- L12: "contracted, virus, contagious" (cold pole)
- L28: "fever, coughing, cough, flu" vs "proudly, reel, bait, trout"

**"The bank was steep and" vs "The bank was closed and"** (noun disambiguated by
adjective):
- L12: "climb, inclined, steep" (terrain pole)
- L27-28: "bank, banks, banking" reappear in the contrastive projection —
  the noun's representation is revisited after the modifier has disambiguated
  it. This is consistent with a two-pass pattern (resolve meaning from the
  modifier, then update the noun representation), though the contrastive
  projection reads content, not mechanism.

**"He was fired up and" vs "He was fired and"** (particle changes meaning):
- L28: "excited, ready, energetic" (fired up) vs "sued, blacklist, lawsuits"
  (fired)

---

## 5. Replicating landmark findings

### 5.1 Indirect object identification

The IOI circuit (Wang et al. 2023) identifies how GPT-2 resolves which name a
pronoun refers to. We replicate the core finding with the contrastive method.

**Design:** Contrast "John and Mary went to the store. John gave a book to" vs
the same with names swapped. The IO name should appear in the contrastive
projection at the prediction site.

**Prompts (example):**
- A: "John and Mary went to the store. John gave a book to" → Mary
- B: "Mary and John went to the store. Mary gave a book to" → John

**Contrastive trajectory:**
- L8: gender emerges ("himself, his" vs "herself, her")
- L24: IO names appear ("Mary, Mary" vs "John, John")
- L28: fully crystallized ("Mary, Mary, mary" vs "John, John, john")

**Per-head decomposition at L24:** Decomposing the attention output by head
identifies which heads carry the IO name. Three heads dominate consistently
across three name pairs (John/Mary, Alice/Bob, Dan/Eve):

| Head | John/Mary (norm) | Alice/Bob | Dan/Eve | Content |
|------|-----------------|-----------|---------|---------|
| H14 | 8.3 | — | 11.4 | IO name (MD/Mary, de/draw) |
| H1 | 7.5 | 5.9 | 7.0 | IO name (Mary, Bob, ever/ves) |
| H16 | 3.4 | 4.5 | 12.8 | IO name (Mary, Bob, Eve) |

These heads have the largest contrastive norms at L24 and read IO-name tokens
when projected through W_U. No other head exceeds norm 3.0 consistently. This
identifies candidate name-mover heads — the same functional role that Wang
et al. found in GPT-2 via circuit analysis, here located in Phi-2 using only
contrastive projection. Causal
verification (e.g., ablating these heads) would be needed to confirm they are
necessary for the computation.

**Accuracy:** 30/30 across 5 name pairs × 3 templates on Phi-2 (100%),
30/30 on Pythia-1.4B (100%), 29/30 on Pythia-410M (97%).

### 5.2 Factual recall

**Design:** Contrast prompts requiring different factual answers.

**"The Eiffel Tower is located in" vs "The Colosseum is located in":**
- L20: "French, France" vs "ancient, Roman, Alexandria"
- L28: "French, Paris" vs "Rome, Roma, Gladiator"

**Per-head decomposition:** At L24, H13 carries the largest contrastive norm
(3.9, reads "France, French, Paris"). At L28, the signal distributes across
multiple heads (H21, H7, H0), consistent with the factual content spreading
from a concentrated source to a broader representation — though this is a
single case and the spreading pattern is not causally verified.

**Different factual domains:** "The capital of France is" vs "The capital of
Japan is" — the contrastive reads "Paris, France, French" on the France pole
and "Japanese, Japan, Tokyo" on the Japan pole. Each entity's associated
knowledge cluster appears on its respective side.

### 5.3 Factual recall vs hallucination

When the model hallucinates — producing a confident but fabricated answer for
a fictional entity — the contrastive projection reveals what the model has
retrieved: specific facts for real entities, and nothing beyond name fragments
for fictional ones.

**Design:** We construct matched pairs where one prompt elicits genuine recall
and the other elicits hallucination, keeping the frame identical:

| Real prompt | Prediction | Fictional prompt | Prediction |
|-------------|-----------|-----------------|-----------|
| Nikola Tesla, born in 1856, invented the | Tesla coil... | Ludvig von Vogelkirche, born in 1859, invented the | first practical electric motor... |
| Marie Curie, born in 1867, discovered | the elements polonium and radium... | Helena Brandström, born in 1871, discovered | a new species of moth... |

Both sides produce confident, specific answers. But the contrastive projection
at L28 reads different content on each pole:

| Pair | Real pole (L28) | Fictional pole (L28) |
|------|----------------|---------------------|
| Tesla vs Vogelkirche | Tesla, alternating, Altern, electric | Vog, v, von, Von |
| Curie vs Brandström | radio, Radio, Radiation, radioactive | Brand, M, H, a |

The real pole reads *factual associations* (Tesla → alternating current,
Curie → radioactivity). The fictional pole reads *name fragments* (Vog, von,
Brand) — the model has retrieved nothing about the entity beyond the tokens
of its name and their cultural associations.

**Confirmation via entity-vs-generic contrast.** Contrasting each entity
against a bare frame ("A person, born in 1856, invented the") isolates what
the entity name adds. Tesla minus generic reads "Tesla, alternating, AC" at
L28 — specific factual content. Vogelkirche minus generic reads "Vog, von,
Von" — only name tokens. The hallucinated entity's representation contains
no factual content beyond the name itself.

**Contrastive norm.** The relative norm ||Δh||/||h|| at L28 is systematically
larger for real-vs-fictional pairs (mean 0.98) than real-vs-real pairs (mean
0.70). This is partly because fictional entities lack specific content to
share with the real entity, and partly because the entities are less similar.

**Entropy.** Real entities predict with lower entropy (H = 1.5–2.5) than
fictional ones (H = 3.4–6.3), consistent with the model having specific
knowledge to draw on. But entropy alone cannot distinguish "confident recall"
from "confident hallucination" — the mountain case illustrates this: Mount
Silverhorn (fictional) predicts "2,856 meters" with H = 2.8, similar to
Mount Everest's H = 1.5. The contrastive projection shows the difference:
Everest's pole reads "Nepal, Tibet, Himal" (geographic knowledge), while
Silverhorn's reads "1300, 1100, 1200" (number-range priors calibrated to
"Southern Alps," not specific factual recall).

The contrastive projection does not detect a "hallucination flag." It reads
what the model has retrieved, and for hallucinated entities, the retrieval is
empty of factual content — only name tokens and contextual priors remain.

### 5.4 Successor heads and temporal structure

All successors are predicted correctly, including wrap-arounds:
"After Saturday comes" → Sunday; "After Sunday comes" → Monday;
"After December comes" → January.

**Contrastive trajectory:** Contrasting consecutive successors (e.g., "After
Monday comes" vs "After Tuesday comes"), the contrastive projection at L24
reads the input day name on its respective pole ("Monday" tokens positive,
"Tuesday" negative). By L28 the successor day appears: "Tuesday" on the
Monday pole, "Wednesday" on the Tuesday pole. The successor computation is
visible as the transition from input-day to output-day content across layers.

**Per-head decomposition at L28:** Decomposing the contrastive signal by
attention head identifies H11 as the dominant successor head. Its contrastive
norm reveals a discontinuity at the weekend boundary:

| Day pair | H11 norm | Next-largest head |
|----------|----------|------------------|
| Mon→Tue | 2 | H20 (3) |
| Tue→Wed | 3 | H20 (3) |
| Wed→Thu | 3 | H20 (2) |
| Thu→Fri | 4 | H20 (5) |
| Fri→Sat | 4 | H25 (3) |
| **Sat→Sun** | **11** | H12 (4) |
| **Sun→Mon** | **11** | H12 (5) |

H11's norm triples at the weekend boundary. The same head dominates
month-pair contrasts at L28, with norms 3-9 across all twelve transitions.

H11 dominates the successor signal but does not carry it alone — H20 and H25
contribute at comparable norms for some day pairs (e.g., H20 norm=5 at
Thu→Fri vs H11 norm=4).

A separate PCA analysis of the raw hidden states (not using the contrastive
projection) finds circular geometry for months and days; this is reported in
the supplementary materials as it does not use the paper's method.

---

## 6. Contrastive axis taxonomy

Using 2×2 factorial designs (crossing two binary axes, e.g., past/future ×
happy/sad), we measure whether each axis produces a consistent contrastive
direction across content. Consistency is the cosine between the axis direction
extracted from two different content fillers. We test 15 axes across four models
(Pythia-410M, Pythia-1.4B, Phi-2, Phi-4).

### 6.1 Three tiers of axis consistency

Axes partition into three tiers:

**Tier 1: Universal, token-readable** (cos > 0.7 in all four models).
Eight axes exceed 0.7 consistency in all four models, though some weaken at
scale: code/natural language (0.88–0.95), positive/negated (0.84–0.90),
past/future (0.81–0.89), English/French (0.83–0.98, declines at scale),
assignment/equality-test (0.79–0.92, improves with scale),
CAPS/lowercase (0.71–0.87), doubt/certainty (0.71–0.96, weakens at scale),
claim/question (0.74–0.83). These axes are near-orthogonal to their content fillers
(mean |cross-cosine| 0.06–0.14).

**Tier 2: Partially consistent** (cos 0.5–0.8 in some models).
Four axes depend on model family or content: formal/informal (strong in
Pythia, weak in Phi), thought/speech (moderate, reads as
subjective-evaluation vs reporting), cause/effect (weakens at scale).

**Tier 3: Not a direction** (cos < 0.5 in all models).
Four axes never form a consistent direction: animate/inanimate (0.34–0.48),
active/passive (0.17–0.65, entangles with content), literal/metaphorical
(0.19–0.35, worsens at scale), salient-entity/generic (collapses to −0.12
at Phi-4).

### 6.2 Negation is not cancellation

Projecting different negation types onto the contrastive `not` direction:

| Negation | Pythia-1.4B | Phi-2 |
|----------|-------------|-------|
| not | +43 | +69 |
| never | +36 | +63 |
| no longer | +35 | +54 |
| **not never** | **+35** | **+66** |
| rarely | +22 | +49 |

Double negation ("not never") does not cancel — it projects at 81–95% the
strength of single "not." The model represents "not never" as emphatic
negation. Each negation type has its own token readout: `not` reads as
"not, NOT, Not"; `no longer` reads as "gone, now, replaced" (temporal
displacement); `rarely` reads as "seldom, usually, often" (frequency scale).

### 6.3 Metaphor is processed by domain routing, not a flag

The literal/metaphorical axis has the lowest consistency (0.19–0.35) because
metaphor is not a single direction. Instead, each metaphorical use routes to
its target domain:

| Contrast | Literal pole reads | Metaphor pole reads |
|----------|-------------------|-------------------|
| cold: ice vs reception | temperatures, 32°F | tense, gloomy, awkward |
| sharp: knife vs criticism | blade, stainless | sarcastic, hostile |
| bright: lamp vs student | blinding, illuminating | gifted, intellectual |

The contrastive projection shows that the model processes metaphor by
activating domain-specific tokens at mid-layers (L16–24), not by toggling a
figurativity feature. This explains why metaphor does not form a linear axis
This suggests (but does not test) that probing classifiers trained on one
metaphor domain would not transfer to another.

---

## 7. Ordering mechanism

Phi-2 solves 3-entity transitive ordering problems ("Alice is taller than Bob.
Bob is taller than Carol. Who is the shortest?") with 100% accuracy across 22
variations (different names, properties, premise orders, distractors) when
querying endpoints (tallest/shortest). The mechanism does not extend to
middle-position queries on longer chains — with 5 entities, the model cannot
identify the 2nd or 3rd tallest from pairwise comparisons without first
generating the sorted list (see supplementary).

**Method:** We contrast all 6 permutations of the 3-entity ordering against
each other and project through W_U at the final premise position and the
answer position. For scale-invariance, we contrast the same ordering across
different properties (e.g., "taller" vs "richer" with the same name
assignments) and measure cosine similarity of the contrastive difference
vectors.

**Representation:** At the chain-completion position, the contrastive
projection reads the bottom-of-chain entity as the top logit. The
representation is scale-invariant: contrasting "Alice is taller than Bob" vs
"Bob is taller than Alice" and the same pair with "richer" yields cosine > 0.98
between the two contrastive vectors. PCA of the 6 permutations' raw hidden
states (not using the contrastive projection) reveals a 2D structure (SVD:
60% + 28%) where orderings sharing the same bottom entity cluster together.

**Query mechanism:** Contrasting "Who is the shortest?" vs "Who is the
tallest?" at the question position reads a semantic direction. At the answer
position (L24), this direction selects the correct endpoint from the ordering
representation.

**Multi-scale limitation:** When two independent orderings are present
(richness + height), the scale-invariant mechanism confuses them. The
contrastive projection shows the model blending the two orderings rather than
maintaining them separately.

**Scaling:** Pythia-410M and 1.4B show no ordering signal in the contrastive
projection (only name/position bias). The mechanism emerges between 1.4B and
2.7B.

---

## 8. Discussion

### What the method is

A training-free exploratory tool for reading how two inputs differ in the
residual stream, at each layer and position, in token space. One matrix
multiply per layer-position pair on a difference vector. The method locates
where a distinction first appears (per-position reading), which component
writes it (attention vs MLP decomposition), and which head carries it
(per-head decomposition). It provides the "where and when" of a computation,
identifying targets for causal verification.

### What the method reads

The prediction-shaped component of the representation — content aligned with
W_U's token rows. This is continuation preparation: tokens the model has
prepared for possible use at current or future positions, modulated by context.

### What the method does not read

Non-prediction-shaped computation. Structure in the residual stream that is not
aligned with W_U is invisible to the projection. The method's silence on a
direction does not imply the direction is absent — only that W_U cannot decode
it.

### Relationship to circuit analysis

The contrastive projection locates phenomena; circuit analysis explains them.
Wang et al. (2023) mapped the full IOI computational subgraph in GPT-2,
identifying specific heads (e.g., S-inhibition heads, name-mover heads) and
their causal roles. Our method identifies heads with the same functional
signature in Phi-2 (H14, H1, H16 at L24) by contrastive norm, but does not
establish their causal roles — that requires
activation patching or path patching. The method is closest to causal tracing
(Meng et al. 2022): it identifies where information concentrates, then hands
off to heavier tools for causal verification.

### Limitations

- **Curated pairs, not sampled.** All demonstrations use hand-constructed
  minimal pairs.
- **LayerNorm bypassed.** We skip the final LayerNorm, so W_U receives
  vectors at the wrong scale. Token rankings are empirically invariant to this
  choice, but raw contrastive norms are not comparable across layers due to
  residual-stream norm growth.
- **W_U readability not guaranteed.** The difference of two states was never
  trained for W_U projection. Token labels at intermediate layers are W_U's
  nearest-neighbour assignments, not verified names for model computations.
- **Smoothness is not W_U-specific.** Trajectory coherence (consecutive-layer
  cosine) is a property of Δh, not of W_U. This metric does not validate that
  W_U reads anything meaningful.
- **Exploratory, not causal.** The per-position trace suggests information
  flow paths; the per-head decomposition identifies large contributors. Neither
  establishes causality. Activation patching is required to confirm that a
  component is necessary, not merely correlated. We verify causality via
  activation patching for the hot dog case (§4.1) and via injection recovery
  for four cases (§3.1), but the per-head decomposition (§5) is observational.
- **Tokenization for per-position reading.** The per-position trace (§2.2) requires that the read position corresponds to the
  same structural role in both inputs — e.g., "dog" must be at the same
  position in both prompts. This is naturally satisfied by minimal pairs that
  differ in one token.
- **Model coverage.** Primary mechanistic results (§4) on Phi-2 only. IOI
  replicates on Pythia models. The axis taxonomy (§6) covers four models
  (Pythia-410M, Pythia-1.4B, Phi-2, Phi-4) and shows both universal and
  model-family-dependent patterns.

---

## 9. Related work

**Contrastive activation methods.** RepE (Zou et al. 2023), ActAdd (Turner et
al. 2023), CAA (Rimsky et al. 2024) use matched-pair subtraction for steering.
Du et al. (2026) apply it to R1-style reasoning models. We use the same
arithmetic for reading, with per-position tracing and attention/MLP
decomposition.

**Logit and tuned lens.** nostalgebraist (2020), Belrose et al. (2023). Project
individual states through W_U. The contrastive projection reads the content
that differs between two inputs — a different subspace from what the logit lens
shows for either input individually.

**Circuit analysis and causal tracing.** Wang et al. (2023) reverse-engineered
the IOI circuit in GPT-2 via path patching. Meng et al. (2022) used causal
tracing to localize factual associations. Gould et al. (2024) identified
successor heads via attention pattern analysis. Our method recovers the key
observational findings from these papers (which layers, which heads, which
content) but does not establish causality — it is an exploratory complement
to these causal techniques. For successor heads, the per-head contrastive
decomposition additionally reveals the discontinuity structure at temporal
boundaries (§5.3).

---

## Acknowledgements

This research was conducted with Claude (Anthropic) as a collaborative tool. The
human author directed all research questions, validated all claims, and takes
full responsibility. Code and data at [REPO].

---

## References

Belrose, N., et al. (2023). Eliciting latent predictions with the tuned lens.
Du, Y., et al. (2026). From latent signals to reflection behavior.
Gould, S., et al. (2024). Successor heads.
Meng, K., et al. (2022). Locating and editing factual associations in GPT.
nostalgebraist. (2020). interpreting GPT: the logit lens.
Rimsky, N., et al. (2024). Steering Llama 2 via CAA.
Turner, A., et al. (2023). Activation addition.
Wang, K., et al. (2023). Interpretability in the wild: IOI circuit.
Zou, A., et al. (2023). Representation engineering.
