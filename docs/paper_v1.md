# Contrastive Projection: A Training-Free Probe for Reading Transformer Internals

**Olli Tuomi**, [Affiliation]

---

## Abstract

We describe a training-free method for reading how a transformer's internal
representation differs between two inputs. Subtracting hidden states at each
layer and projecting through the unembedding matrix W_U yields a contrastive
trajectory — a layer-by-layer readout of what separates the two residual
streams in token space. The method requires one matrix multiply, no probes to
train, and no learned parameters.

We demonstrate the method on Phi-2 (2.7B) across four domains:

**Lexical disambiguation.** "The hot dog was" vs "The cold dog was" — the
contrastive projection traces compound-noun recognition to a specific position
and layer. At the "dog" position, food vocabulary ("fried") first appears at L5.
Attention weights confirm a two-hop chain: "dog" reads "hot" at L0, then "was"
reads the compound from "dog" at L5-6. Activation patching at the "dog"
position eliminates the food signal entirely.

**Indirect object identification.** "John and Mary went to the store. John gave
a book to" — the IO name (Mary) emerges at L24 in the contrastive projection.
Tested on 30 name/template combinations: 100% correct on Phi-2, 100% on
Pythia-1.4B, 97% on Pythia-410M.

**Factual recall.** "The Eiffel Tower is in" vs "The Colosseum is in" — the
contrastive reads "Paris, France" vs "Rome, Roma" at L24-28. When contrasted
against fictional entities ("The Crystal Palace of Zoria"), the factual content
separates cleanly from the fictional confabulation.

**Circular successor representation.** "After January comes" for all twelve
months: the hidden states at L28 form a near-perfect circle in 2D PCA, with
angular steps averaging 30° (= 360°/12). Days of the week show the same
circular structure at ~51° steps (= 360°/7), with larger jumps at the
workweek/weekend boundary. This replicates, via a training-free method, the
Fourier features previously found only through circuit analysis of grokked
models.

The contrastive projection reads the prediction-shaped component of the
representation — content aligned with W_U's token space. Trajectory smoothness
is a property of the difference vector Δh, not of W_U: random projections give
identical consecutive-layer cosine (0.962 vs 0.962). The method is silent on
non-prediction-shaped computation. We release code and data.

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
revealing the content that separates the two inputs but that neither
constituent's logit lens can see.

The probe is mechanical: given two inputs, subtract their hidden states at each
layer and project the difference through W_U. The top tokens at each layer are
the trajectory. We make two claims:

1. **Descriptive.** The difference between two matched inputs, projected through
   W_U at each layer and position, reads coherent token-space content that
   changes across layers. The content is set by the chosen pair; the
   model-relevant observations are the layer-wise dynamics and the per-position
   information flow.
2. **Mechanistic.** By reading the contrast at every position, we can trace
   which position first computes a meaning distinction, which component
   (attention or MLP) contributes it, and how information flows between
   positions. This replicates, without training, findings that previously
   required circuit analysis.

All primary results use Phi-2 (Microsoft, 2.7B parameters, 32 layers).
Cross-model replication on Pythia-410M and Pythia-1.4B is reported for IOI.

---

## 2. Method

### 2.1 Contrastive projection

Given two inputs *c* and *k*:

1. Run both and extract hidden states at every layer at the read position:
   h_c[L] and h_k[L] for L = 0, …, N.
2. Compute Δh[L] = h_c[L] − h_k[L].
3. Project: logits[L] = Δh[L] · W_U^T.
4. Read the top-K tokens (the contrast in token space).

No parameters are fit. The choices are the input pair, the read position, and K.

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

Trajectory smoothness (consecutive-layer cosine) is a property of Δh drifting
gradually in the residual stream, not a property of W_U. Random projections of
the same shape give identical smoothness (mean cosine 0.962 for both W_U and
random at L9+). The interpretable token labels are W_U's contribution; the
smoothness is not.

The method is silent on non-prediction-shaped computation — structure in the
residual stream that is not aligned with W_U's token rows.

---

## 3. Lexical disambiguation

### 3.1 Compound noun: hot dog

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

**Activation patching:** Replacing the "dog" position's hidden state (L0) with
the cold-dog version eliminates the food signal entirely: P(food tokens) drops
from 12% to 0.03%. Replacing the "hot" position's state has minimal effect
(food stays at 69-83%) — the information has already been copied to "dog" via L0
attention.

**Mechanism:** Attention at L0 copies "hot" to the "dog" position. The compound
is recognized at the "dog" position by L5 ("fried" first appears). Attention at
L5-6 reads this from "dog" to "was." The model's food/animal disambiguation is
a two-hop attention chain: hot→dog (L0), dog→was (L5).

### 3.2 Other disambiguation cases

The same method traces disambiguation in verb-object pairs and noun ambiguity:

**"He caught a cold and" vs "He caught a fish and"** (verb meaning changes):
- L12: "contracted, virus, contagious" (cold pole)
- L28: "fever, coughing, cough, flu" vs "proudly, reel, bait, trout"

**"The bank was steep and" vs "The bank was closed and"** (noun disambiguated by
adjective):
- L12: "climb, inclined, steep" (terrain pole)
- L27-28: attention reads back "bank, banks, banking" — re-reading the
  ambiguous noun after disambiguation. A two-pass pattern: resolve meaning from
  the modifier (L5-12), then re-read the noun (L27-28).

**"He was fired up and" vs "He was fired and"** (particle changes meaning):
- L28: "excited, ready, energetic" (fired up) vs "sued, blacklist, lawsuits"
  (fired)

---

## 4. Replicating landmark findings

### 4.1 Indirect object identification

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

**Accuracy:** 30/30 across 5 name pairs × 3 templates on Phi-2 (100%),
30/30 on Pythia-1.4B (100%), 29/30 on Pythia-410M (97%).

The method sees the same phenomenon as Wang et al. — the IO name
crystallizes in the residual stream — without circuit analysis, attention head
identification, or activation patching.

### 4.2 Factual recall

**Design:** Contrast prompts requiring different factual answers.

**"The Eiffel Tower is located in" vs "The Colosseum is located in":**
- L20: "French, France" vs "ancient, Roman, Alexandria"
- L28: "French, Paris" vs "Rome, Roma, Gladiator"

**Factual vs fictional:** "The capital of France is" vs "The capital of Narnia
is" — the model predicts "Paris" and "Cair Paravel" respectively. The
contrastive reads "Paris, France, Berlin" on the factual pole, cleanly
separating real from fictional recall.

**"Einstein developed the theory of" vs "Glorb developed the theory of":**
- L24: "Einstein, Albert, relativity" on the factual pole
- The imaginary entity produces no specific factual content

### 4.3 Circular successor representation

**Design:** Run "After [X] comes" for all twelve months; extract hidden states
at L28; perform PCA.

**Finding:** The twelve months form a near-perfect circle in the top-2 PCA
subspace.

| Month | Angle | Step |
|-------|-------|------|
| January | +98.7° | |
| February | +69.3° | −29.3° |
| March | +43.4° | −25.9° |
| April | +6.8° | −36.6° |
| May | −19.6° | −26.4° |
| June | −33.3° | −13.7° |
| July | −76.3° | −43.1° |
| August | −102.3° | −25.9° |
| September | −137.6° | −35.3° |
| October | −155.1° | −17.5° |
| November | +169.2° | −35.6° |
| December | +146.7° | −22.6° |

Mean angular step: −30.0° (= 360°/12). The progression is monotonically
clockwise through 360°.

Days of the week show the same circular structure at ~51° steps (= 360°/7),
with larger jumps at Thursday→Friday (88°) and Sunday→Monday (74°) — the
workweek/weekend boundary.

All successors are predicted correctly, including the wrap-around:
"After Saturday comes" → Sunday; "After Sunday comes" → Monday;
"After December comes" → January.

The circular structure in W_U-projected space replicates, via a training-free
method, the Fourier features previously found only through weight-level
analysis of grokked models (Nanda et al. 2023, Zhong et al. 2024).

---

## 5. Syntactic and semantic axes

### 5.1 Grammatical features

Contrastive directions extracted from minimal pairs ("The dog was" vs "The dogs
were") reveal consistent syntactic axes:

| Feature | Training cos | Transfer cos | Axis stable? |
|---------|-------------|-------------|-------------|
| Number | 0.62–0.88 | 0.45–0.86 | Yes |
| Tense | 0.50–0.75 | 0.42–0.82 | Yes |
| Gender | 0.40–0.76 | 0.20–0.79 | Moderate |

The three axes are approximately orthogonal (number-tense cos = −0.02,
number-gender cos = +0.20, tense-gender cos = −0.02).

### 5.2 Semantic features

Animacy and valence do not form consistent single directions — each noun pair
reads its own content (dog/rock reads "barking, leash" vs "structure, built,"
not a generic animacy axis).

---

## 6. Ordering mechanism

Phi-2 solves transitive ordering problems ("Alice is taller than Bob. Bob is
taller than Carol. Who is the shortest?") with 100% accuracy across 22
variations (different names, properties, premise orders, distractors).

**Representation:** At the chain-completion position, the bottom-of-chain
entity has the highest logit (rank 1). The representation is scale-invariant
(cos > 0.98 across taller/richer/older/faster/heavier/smarter).

**Query mechanism:** The question word ("shortest" vs "tallest") encodes a
semantic direction at the question position. At the answer position (L24), this
direction selects the correct endpoint from the ordering representation.

**Multi-scale limitation:** When two independent orderings are present
(richness + height), the scale-invariant mechanism confuses them. The model
cannot maintain two orderings simultaneously.

**Scaling:** Pythia-410M and 1.4B have no ordering mechanism (name/position
bias). The mechanism emerges between 1.4B and 2.7B.

---

## 7. Discussion

### What the method is

A training-free readout of how two inputs differ in the residual stream, at each
layer and position, in token space. One matrix multiply on a difference vector.
The method traces information flow (per-position reading), identifies components
(attention vs MLP decomposition), and reads computed answers (answer tokens
appear in the projection).

### What the method reads

The prediction-shaped component of the representation — content aligned with
W_U's token rows. This is continuation preparation: tokens the model has
prepared for possible use at current or future positions, modulated by context.

### What the method does not read

Non-prediction-shaped computation. Structure in the residual stream that is not
aligned with W_U is invisible to the projection. The method's silence on a
direction does not imply the direction is absent — only that W_U cannot decode
it.

### Limitations

- **Curated pairs, not sampled.** All demonstrations use hand-constructed
  minimal pairs.
- **W_U readability not guaranteed.** The difference of two states was never
  trained for W_U projection. Token labels at intermediate layers are W_U's
  nearest-neighbour assignments, not verified names for model computations.
- **Smoothness is not W_U-specific.** Trajectory coherence (consecutive-layer
  cosine) is a property of Δh, not of W_U. This metric does not validate that
  W_U reads anything meaningful.
- **Mechanistic claims require verification.** The per-position trace suggests
  information flow paths; activation patching is needed to confirm causality.
  We verify the hot dog case; other cases are traced but not patched.
- **Single model.** Primary results on Phi-2 only. IOI replicates on Pythia
  models.

---

## 8. Related work

**Contrastive activation methods.** RepE (Zou et al. 2023), ActAdd (Turner et
al. 2023), CAA (Rimsky et al. 2024) use matched-pair subtraction for steering.
Du et al. (2026) apply it to R1-style reasoning models. We use the same
arithmetic for reading, with per-position tracing and attention/MLP
decomposition.

**Logit and tuned lens.** nostalgebraist (2020), Belrose et al. (2023). Project
individual states through W_U. The contrastive projection reads the content
that differs between two inputs — a different subspace from what the logit lens
shows for either input individually.

**Circuit analysis.** Wang et al. (2023) reverse-engineered the IOI circuit in
GPT-2. Nanda et al. (2023) found Fourier features in grokked models. Gould et
al. (2024) identified successor heads. Meng et al. (2022) traced factual
recall. Our method replicates key findings from these papers using a single,
training-free technique.

**Grokking and circular representations.** Nanda et al. (2023) found that
grokked models use Fourier features for modular arithmetic. Zhong et al. (2024)
showed clock and pizza representations. Our finding that month and day-of-week
representations form circles in the W_U-projected space connects to this work,
suggesting that circular temporal structure is present even in large pretrained
models, not only in small grokked ones.

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
Nanda, N., et al. (2023). Progress measures for grokking via mechanistic interpretability.
nostalgebraist. (2020). interpreting GPT: the logit lens.
Rimsky, N., et al. (2024). Steering Llama 2 via CAA.
Turner, A., et al. (2023). Activation addition.
Wang, K., et al. (2023). Interpretability in the wild: IOI circuit.
Zhong, Z., et al. (2024). The clock and the pizza.
Zou, A., et al. (2023). Representation engineering.
