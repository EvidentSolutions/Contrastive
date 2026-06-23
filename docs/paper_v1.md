# Contrastive Projection: Reading Transformer Internals Through Desuperposition

**Olli Tuomi**, Evident Solutions Oy

---

## Abstract

A transformer's residual stream carries multiple signals in superposition.
The logit lens reads their incoherent sum; we subtract matched inputs before
projecting, isolating the axis of variation. This contrastive projection —
one matrix multiply per layer-position pair, no training — reads how two
inputs differ in token space at every layer, producing a trajectory of
the computation that separates them.

We validate in three ways. (1) Injecting the contrastive direction into the
opposing input recovers the prediction gap (z = 223–4249 across four cases).
(2) Dose-response tests confirm the direction, not just the subspace, is
causal. (3) MLP neurons whose fc1 weights detect the same features the
method reads gate selectively on the contrast (GELU pre-activation +1.50 vs
−0.01); across 18 contrasts spanning lexical, semantic, syntactic, and
cross-lingual phenomena, every case produces strictly gated neurons, with
zero neuron reuse between contrasts (67 unique neurons out of 10,240 per
layer).

When a single contrastive pair produces uninterpretable tokens, the signal
is still present but superposed with pair-specific noise. Multi-contrast
triangulation — averaging across multiple baselines — recovers token-shaped
causal content: for "caught a cold," single-pair recovery is 1%, five-baseline
triangulation recovers 77% (reading "doctor, doctors, rest"). Entity-identity
contrasts (names, cities) desuperpose from a single pair; compositional
contrasts (food compounds, moral judgment, quantifier pragmatics) require
multiple baselines.

We apply the method to Phi-2 (2.7B), tracing compound-noun recognition to
a two-hop MLP→attention circuit confirmed by activation patching, replicating
IOI name-mover heads and factual recall concentration, distinguishing real
factual recall from hallucinated name fragments, identifying a weekend
discontinuity in successor heads, and mapping 15 semantic axes across four
models. We release code and data.

---

## 1. Introduction

A transformer processing "The hot dog was" predicts continuations about food.
The same model processing "The cold dog was" predicts continuations about an
animal. The logit lens (nostalgebraist 2020) projects each hidden state through
W_U and at intermediate layers reads the same function words for both —
"not, no, made, a, more." The food distinction is invisible: it is present in
the residual stream but superposed with other signals that dominate the
projection.

Subtracting one hidden state from the other before projecting through W_U
cancels the shared signals and reads what differs. At layers 8–24 for the
hot-dog case, this contrastive projection reads "fried, crispy, delicious,
flavor" — food vocabulary absent from either constituent's logit-lens top-20
(0–1/5 overlap through L24). The subtraction performs desuperposition: it
isolates one axis of variation from the residual stream's superposed content.

The arithmetic is identical to computing a RepE/ActAdd steering vector (Zou
et al. 2023; Turner et al. 2023) and reading it through a logit lens
(nostalgebraist 2020). The novelty is not the subtraction but three things
built on it:

1. **Systematic trajectory reading.** Per-position tracing locates where a
   distinction first appears and how it flows between positions. Per-head
   decomposition identifies which attention heads carry it. Layer-by-layer
   readout tracks how the content changes from early detection to final
   prediction.

2. **Multi-contrast triangulation.** A single contrastive pair may produce
   uninterpretable tokens when the target signal is superposed with
   pair-specific noise. Contrasting the same target against multiple baselines
   and averaging isolates the shared causal direction. This recovers
   token-shaped causal content from cases where single-pair readout fails:
   "caught a cold" recovers 1% from a single pair, 77% from five-baseline
   triangulation (reading "doctor, doctors, rest").

3. **MLP neuron correspondence.** The features the method reads externally
   correspond to features the model detects internally. MLP neurons whose fc1
   weights align with the contrastive input direction gate selectively via
   GELU, and their fc2 columns write the same tokens the contrastive method
   reads. Across 18 contrasts, every case produces strictly gated neurons with
   zero neuron reuse between contrasts.

We apply the method to Phi-2 (2.7B), with cross-model replication on
Pythia-410M, Pythia-1.4B, and Phi-4 (14B).

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

The contrastive projection performs external desuperposition of the residual
stream. At any given layer, the hidden state carries multiple token-shaped
signals in superposition. A raw logit-lens readout sees their incoherent
sum — often dominated by function words or producing uninterpretable token
rankings. The contrastive subtraction cancels signals shared between the two
inputs and isolates the axis of variation, making the readout coherent.

What W_U reports as token labels at intermediate layers are not predictions
but continuation preparations: tokens the model might produce at the current
or future positions, modulated by context.

**Single-pair vs. multi-contrast readout.** A single contrastive pair
isolates one axis of variation but the readout may still contain pair-specific
noise superposed with the shared signal. When the target concept is a single
entity (a name, a city), this noise is small and the readout is clean. When
the target is compositional (a food compound, a moral judgment, a quantifier's
pragmatic force), pair-specific noise can dominate, producing uninterpretable
tokens. Multi-contrast triangulation — contrasting the same target against
several baselines and averaging — cancels pair-specific noise and recovers
the shared causal direction.

We verify this empirically. For "She caught a cold" vs "She caught a fish,"
the top-20 W_U directions of the single-pair Δh recover 1% of the prediction
gap when injected. The same case triangulated against five contrasts (fish,
ball, bus, thief, glimpse) yields a shared direction reading "doctor, doctors,
rest, see" that recovers 77%. The causal content was token-shaped but buried
under pair-specific superposition (Table 1).

| Case | Single-pair recovery | Multi-contrast recovery | Shared direction reads |
|------|---------------------|------------------------|----------------------|
| IOI (Mary/John) | 84% | — (not needed) | Mary, mary |
| Capital (Paris/Berlin) | 97% | — (not needed) | Paris, French |
| Caught a cold | 1% | 77% | doctor, doctors, rest |
| Theft/moral | 4% | 75% | evade, hoped, evasion |
| Hot dog food | 11% | 49% | crispy, charred, cooked |
| Some/all (partitive) | -40% | 101% | some, Some, SOME |

*Table 1. Single-pair W_U projection vs. multi-contrast triangulation.
Entity-identity contrasts (top rows) desuperpose trivially. Compositional
contrasts (bottom rows) require multiple baselines to isolate the
token-shaped causal signal from pair-specific noise.*

W_U contributes the interpretable token labels. Trajectory smoothness
(consecutive-layer cosine ~0.9) is a general property of the residual stream,
not specific to meaningful contrasts: unrelated pairs ("The hot dog was" vs
"Quantum mechanics is") are equally smooth (0.93 ± 0.01, N=40) as minimal
pairs (0.88 ± 0.03, N=6), because any two residual streams diverge gradually.
Random directions give cosine ≈ 0.00 (z = 64–93), confirming Δh is structured
rather than noise, but smoothness alone does not validate content.

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

### 3.2 Dose-response and bidirectional causality

The layer-sweep above injects the full Δh at one layer. We can also extract a
specific contrastive *direction* from multiple pairs and test its causal effect
with controlled magnitude.

**Eatability direction.** We extract a food-compound direction by averaging the
contrastive (hot dog minus X dog) for X ∈ {cold, angry, old, pet, stray} at
the dog position, L4 post. These five directions are mutually consistent
(pairwise cos 0.72–0.84). We inject the mean direction into non-food prompts
at the dog position at varying fractions of its natural magnitude:

| Target prompt | Baseline top-1 | +0.25× | +0.50× | +1.0× |
|---------------|---------------|--------|--------|-------|
| The cold dog was | sh (0.45) | sh (0.10) | tasty (0.06) | tasty (0.06), served (0.05) |
| The angry dog was | barking (0.31) | barking (0.34) | barking (0.35) | cooked (0.05) |

Subtracting the direction from "The hot dog was" reverses the effect:

| Fraction | Top-1 | P(top-1) | Interpretation |
|----------|-------|----------|----------------|
| baseline | too | 0.088 | food item |
| −0.50× | more | 0.122 | weakening |
| −1.00× | pant[ing] | 0.234 | animal |
| −1.50× | pant[ing] | 0.297 | animal (stronger) |

The minimum dose that flips top-1 prediction is 0.30× for cold dog (where
"cold" partially aligns with food) and 0.65× for angry dog (where "angry"
strongly primes the animal reading).

**Truth direction.** We extract a truth/falsity direction by averaging the
contrastive (true statement minus false statement) for five fact pairs (Paris/
France, water/0°, Sun/star, dogs/mammals, Tokyo/Japan) at the prediction
position. Injecting into false statements:

| Target prompt | Baseline | +0.50× truth | +1.0× truth |
|---------------|---------|-------------|-------------|
| Paris is not the capital of France. This statement is | false (0.31) | true (0.29) | true (0.42) |

Subtracting from true statements:

| Target prompt | Baseline | −1.0× truth | −1.5× truth |
|---------------|---------|-------------|-------------|
| Paris is the capital of France. This statement is | true (0.37) | false (0.25) | false (0.32), incorrect (0.13) |

The truth direction has lower cross-pair consistency (mean cos ~0.35) than the
eatability direction (0.72–0.84), reflecting the fact that "what makes Paris-is-
the-capital true" and "what makes dogs-are-mammals true" share less structure
than different food-compound contrasts. Despite this, the direction is
bidirectionally causal for the strongest pairs.

### 3.3 Contrastive features correspond to MLP internal structure

The previous sections establish that the contrastive direction is causal
(§3.1) and specific (§3.2). A separate question is whether the features the
method reads — the token-space content of Δh — correspond to structure the
model itself uses, or are artifacts of projecting through W_U.

We test this by decomposing MLP neurons into read→gate→write components.
Each neuron in the MLP has three parts: its fc1 row (a detector that reads
from the residual stream), the GELU activation (a gate that fires or stays
silent), and its fc2 column (a direction written into the residual stream
when the gate opens). If the features read by the contrastive method are
real model representations, then neurons whose fc1 detectors align with the
contrastive input direction should gate selectively — firing for one input
and not the other — and their fc2 write vectors should project to
interpretable tokens through W_U.

We test this across 18 contrastive cases spanning lexical disambiguation,
emotion, morality, metaphor, factual recall, quantifier semantics, tense,
language identity, register, code modality, disaster type, and entity size.
For each case, we identify "strictly gated" neurons: those with pre-GELU
activation above 0.3 for one input and below 0.05 for the other.

**Every case produces strictly gated neurons.** The count ranges from 1
(capital France/Germany, formal/informal, agent swap) to 11 (literal vs
metaphorical cold), with a mean of 3.7 per contrast.

**Example.** At L20, neuron 7828's fc1 row projected through W_U reads
"food, edible, delicious, flavorful, Foods." Its pre-GELU activation is
+1.50 for "The hot dog was" and −0.01 for "The hot cat was." When the gate
opens, its fc2 column writes "flavors, tasting, flavor, flavorful, edible"
into the residual stream. The contrastive method at L20 reads food tokens
from Δh; neuron 7828 internally detects and gates on the same feature.

| Neuron | Layer | Contrast | fc1 reads | Pre-GELU (A/B) | fc2 writes |
|--------|-------|----------|-----------|----------------|------------|
| 7828 | L20 | food compound | food, edible, delicious | +1.50 / −0.01 | flavors, tasting, edible |
| 2133 | L20 | animal size | bulky, larger, cumbersome | +1.07 / −0.04 | larger, bigger, bulky |
| 841 | L28 | IOI gender | himself, his, His | −0.00 / +0.35 | his, himself, His |
| 2226 | L20 | English/French | [French tokens] | −0.04 / +3.40 | ét, dé, ère |
| 5082 | L24 | literal/metaphor | dwindle, skyrocket, grows | +3.01 / −0.05 | rising, rose, increased |
| 925 | L20 | food compound | food, food, foods | +1.00 / −2.31 | [food-associated] |

*Table 2. MLP neurons whose fc1 read direction aligns with the contrastive
input and whose GELU gates selectively. Each neuron detects the same feature
the contrastive method reads from Δh.*

**Zero neuron reuse across contrasts.** Across 49 same-layer pairwise
comparisons between the 18 cases, no strictly gated neuron appears in more
than one contrast. The 18 contrasts activate 67 unique strict neurons out of
10,240 per layer (0.65%). Each contrast activates its own private set of
detectors.

This provides a third form of validation: the contrastive method reads
features that the model's own MLP neurons detect and gate on. The causal
injection tests (§3.1) show the subspace matters; the dose-response tests
(§3.2) show the direction matters; the neuron correspondence shows the
features themselves match the model's internal detector structure.

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
- L4 (post-MLP): "fried" first appears — compound meaning recognized by MLP
- L5 (post-attention): "fried" persists — L5 attention adds little locally
- L24: "fried, seasoning, Flav, Serv, vendor" — full food vocabulary
- L32: "vendor, vendors, stand, topping" — hot dog stand

Sub-layer decomposition at L4–L5 confirms the compound is recognized by the
MLP at L4, not by attention. The MLP contrastive norm at L4 (20.0) dominates
the attention norm (3.8); "fried" appears in the MLP output but not the
attention output. L5 attention writes orthogonally to the food direction
(cos(Δpre, Δattn) < 0.08 across all contrasts tested).

At position 3 ("was") — reads from "dog":
- L5 (attention): "delicious, substitutes, cooked" — food signal arrives via
  attention from "dog" (one layer after MLP recognizes the compound at "dog")
- L6: "fried, delicious, tasty, breakfast"
- L28: "tasty, charred, crispy, delicious, spicy" (hot) vs "grooming, shudder, shaking" (cold)

**Attention routing at L5, position "was":**
Head 19 attends to "dog" position with weight 0.911 in the hot-dog context vs
0.735 in cold-dog. Head 29 attends 0.345 vs 0.138. These heads read the
compound-noun information from "dog" and write it to "was."

**Multi-contrast convergence:** The food-compound direction is stable across
reference points. Five different contrasts (hot dog minus cold/angry/old/pet/
stray dog) produce pairwise cosine 0.72–0.84 at the dog position, confirming
the signal is about food-compound identity, not temperature or emotion.

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

**Mechanism:** Attention at L0 copies "hot" to the "dog" position. The MLP at
L4 recognizes the compound at the "dog" position ("fried" first appears in the
MLP output). Attention at L5 broadcasts this from "dog" to "was" (H19,
attn=0.91). The contrastive projection identified each stage; activation
patching at three positions and multiple layers confirms the information flow.

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

**Tier 1: Cross-family, token-readable** (cos > 0.7 in all four models).
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
figurativity feature. This explains why metaphor does not form a linear axis.
This suggests (but does not test) that probing classifiers trained on one
metaphor domain would not transfer to another.

### 6.4 Axis consistency predicts causal potency

The tier classification connects to the causal tests in §3.2. The
positive/negated axis (Tier 1, cos 0.84–0.90) produces a direction that
bidirectionally flips true/false predictions when injected. The eatability
direction — extracted by the same multi-contrast averaging used for axes —
flips food/animal predictions at 30% of natural magnitude.

We tested whether axis consistency predicts whether the W_U readout of an
axis functions as a frame-forcing token. For six contrastive scenarios, we
extracted the axis direction, read its top tokens through W_U, and tested
those tokens as adjective modifiers on 15 nouns:

| Scenario | Axis consistency | W_U top token | Override rate |
|----------|-----------------|---------------|---------------|
| flying (vs parked) | 0.74 | "future", "enabled" | 87–93% |
| stolen (vs displayed) | 0.67 | "rightfully" | 93% |
| deadly (vs harmless) | 0.57 | "deadly" | 100% |
| burning (vs standing) | 0.59 | "got", "lis" | 0–7% |
| frozen (vs fresh) | 0.44 | "oop", "paradox" | 7% |

When axis consistency exceeds ~0.6, the W_U readout surfaces tokens that
function as frame-forcers when used as modifiers. Below ~0.5, the direction is
still causal when injected (the burning direction shifts predictions from
"built" to "evacuated") but its W_U readout does not produce usable tokens.
The contrastive direction is causally relevant in both cases; only its
token-readability depends on consistency.

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

### What the method does

Contrastive projection desuperposes the residual stream along a chosen axis
of variation. The subtraction cancels signals shared between two inputs;
the W_U projection reads the remainder in token space. The method locates
where a distinction first appears (per-position reading), which component
writes it (attention vs MLP decomposition), and which head carries it
(per-head decomposition).

### What the readout means

The contrastive projection reads continuation preparations: token-shaped
content the model has written into the residual stream for use at current
or future positions. This content is aligned with W_U's token rows. The
token labels are W_U's nearest-neighbour assignments to directions in the
residual stream — names for what the model is computing, verified by three
independent tests: causal injection (§3.1), dose-response (§3.2), and MLP
neuron correspondence (§3.3).

When the readout produces uninterpretable tokens, the target signal is
typically still present but superposed with other signals. Multi-contrast
triangulation (§2.4) recovers the causal content in every case we tested.
This does not guarantee that all model computation is token-readable — it
establishes that for the 18 contrasts tested, the causal content is
token-shaped once properly desuperposed.

### Superposition and contrast design

The quality of the readout depends on the contrast. A well-chosen pair
that varies one axis produces a clean readout. A poorly chosen pair — or
one where the target concept is compositional and superposed with
pair-specific content — produces tokens that reflect the superposition,
not the target. Multi-contrast averaging addresses this for cases we tested,
but we have not established how many baselines are sufficient in general,
nor whether all model computations can be desuperposed by this technique.

### Token readability and causal relevance

Across 16 contrastive pairs, 4 layers, and 32 attention heads (1774
head-level measurements), heads whose contrastive output reads as real
words have higher causal alignment on average (mean |cos| 0.044 for
5/5-readable vs 0.025 for 0/5-readable; Pearson r = +0.15). The most
causally aligned individual head contributions in our sample are
token-readable, and no unreadable head exceeds |cos| = 0.16. The sample
is limited and the correlation is modest.

### Relationship to circuit analysis

The contrastive projection locates phenomena; circuit analysis explains
them. Wang et al. (2023) mapped the IOI circuit in GPT-2 via path
patching. Our method identifies heads with the same functional signature
in Phi-2 (H14, H1, H16 at L24) by contrastive norm, but does not
establish their causal roles. The method is closest to causal tracing
(Meng et al. 2022): it identifies where information concentrates, then
hands off to heavier tools for causal verification.

### Limitations

- **Curated pairs, not sampled.** All demonstrations use hand-constructed
  minimal pairs. The multi-contrast triangulation uses hand-selected
  baselines.
- **LayerNorm bypassed.** We skip the final LayerNorm, so W_U receives
  vectors at the wrong scale. Token rankings are empirically invariant to
  this choice, but raw contrastive norms are not comparable across layers
  due to residual-stream norm growth.
- **W_U readability not guaranteed.** The difference of two states was
  never trained for W_U projection. Token labels at intermediate layers
  are W_U's nearest-neighbour assignments. The MLP neuron correspondence
  (§3.3) provides independent evidence that these labels match internal
  model structure for the cases tested.
- **Smoothness is not W_U-specific.** Trajectory coherence
  (consecutive-layer cosine) is a property of Δh, not of W_U.
- **Exploratory, not causal.** The per-position trace and per-head
  decomposition identify large contributors, not causes. We verify
  causality via activation patching for the hot dog case (§4.1) and via
  injection recovery for four cases (§3.1), but the per-head
  decomposition (§5) is observational.
- **Desuperposition coverage.** We tested multi-contrast triangulation on
  6 cases and MLP neuron correspondence on 18 cases. We do not know
  whether all model computations can be desuperposed by contrastive
  subtraction, nor how many baselines are sufficient in general.
- **Per-position reading requires tokenization alignment.** The read
  position must correspond to the same structural role in both inputs.
- **Model coverage.** Primary mechanistic results (§4) on Phi-2 only.
  IOI replicates on Pythia models. The axis taxonomy (§6) covers four
  models and shows both consistent and model-dependent patterns.

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

**Superposition and sparse autoencoders.** Elhage et al. (2022) characterized
superposition in toy models. Bricken et al. (2023) and Templeton et al. (2024)
use sparse autoencoders to decompose superposed representations into
monosemantic features. Our multi-contrast triangulation achieves a related
decomposition — isolating one signal from superposition — using paired inputs
rather than learned dictionaries. The MLP neuron correspondence (§3.3) connects
to the neuron-level analysis in this literature but uses contrastive gating
rather than unsupervised feature discovery.

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
