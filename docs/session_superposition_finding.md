# Session finding: Contrastive projection as external desuperposition

Date: 2026-06-23

## Core finding

Contrastive subtraction performs the same operation externally that MLPs
perform internally: desuperposition. The residual stream at any given layer
carries multiple token-shaped signals in superposition. A linear readout
(logit lens, raw W_U projection) sees an incoherent sum — "garbage tokens."
The contrastive method cancels shared signals and isolates one direction,
making the readout coherent.

## Evidence chain

### 1. Single-pair W_U projection misidentifies causal content

For "caught a cold" vs "caught a fish," the top-k W_U directions of Δh
recover only 0.6% of the prediction gap. The residual (everything NOT
in the top-k W_U directions) recovers 72%. This was initially interpreted
as "the causal content is not token-shaped."

### 2. Multi-contrast triangulation recovers token-shaped causal content

Contrasting "caught a cold" against five baselines (fish, ball, bus, thief,
glimpse), the *shared* direction reads as "doctor, Doctor, doctors, rest,
see" and recovers 77%. The token-shaped signal was there all along — buried
under pair-specific noise in the single-pair case.

The same reversal occurs for:
- Theft/moral: single-pair token recovery 4% → multi-contrast shared 75%
  (reads as "evade, hoped, hope, evasion, Escape")
- Hot dog: single-pair 11% → multi-contrast shared 49%
  (reads as "crispy, overcooked, juicy, charred, tasty, cooked, grilled")

### 3. Entity-identity contrasts desuperpose trivially

IOI (Mary/John), capital (Paris/Berlin), agent swap (cat/dog) — the causal
content is already a single token, so even single-pair W_U projection works
(84-97% recovery). These cases have minimal superposition to begin with.

### 4. Compositional signals require more contrasts to desuperpose

Food-compound recognition, moral judgment, quantifier semantics — the
causal content is a *relationship* between tokens, not a single token.
It stays superposed longer and needs more contrasts to triangulate.

### 5. The "garbage tokens" are superposition artifacts

Tokens like "$, oli, ((, aria, vine, subscribe" appear repeatedly on the
positive pole of quantifier contrasts. They are not noise — they are the
linear shadow of multiple meaningful signals sharing a subspace. When the
contrast is well-chosen (e.g., partitive mean: most/few/many/half/three),
the readout cleans up to "some, Some, SOME, possibly" (101% recovery).

### 6. Some/all decomposes into two independent signals

- **Partitive mean** (most/few/many/half/three vs some): reads "some, Some,
  SOME, possibly" — recovers 101%. Token-shaped.
- **Universal mean** (all/none/everybody/nobody vs some): reads garbage on
  positive pole, "everyone, everybody, nobody" on negative — recovers -27%
  (harmful). This signal opposes "totality" but the "some-ness" pole is
  distributed across tokens, not yet desuperposed.
- **Cross-content mean** (6 noun phrases): reads "Others, Other, another"
  vs "all, everyone, none, except, everybody" — both poles token-shaped.

### 7. Hot dog vs hot chocolate = food-type routing, not taste

The contrast reads "grill, sausage, burger, BBQ, bun" vs "cocoa, drink,
beverage, brewed, espresso, coffee." Cosine with salty/sweet axis = 0.18.
This is food-preparation-domain routing, same mechanism as metaphor routing
(§6.3 of the paper). The model routes to different culinary neighborhoods.

### 8. MLP neurons as read→gate→write detectors

Individual MLP neurons implement read→gate→write: fc1 row detects an
input pattern, GELU thresholds, fc2 column writes to the residual stream.

Verified at L20: neuron 7828's fc1 reads "food, edible, delicious" from
the residual stream. Pre-GELU: +1.50 for "hot dog" vs −0.01 for "hot cat"
(gates ON/OFF). fc2 writes "flavors, tasting, flavorful, edible."

Sweep across 18 contrasts (food, emotion, morality, metaphor, factual,
quantifier, tense, language, register, code, disaster, size): every case
produces 1–11 strictly gated neurons (mean 3.7, 67 total across 18
contrasts out of 10,240 per layer). Ablating strict neurons produces
10–650× larger KL divergence than ablating the same number of random
neurons.

Additional examples:
- N2133 (L20): reads "bulky, larger, cumbersome" → writes "larger, bigger, bulky"
- N841 (L28): reads "himself, his, His" → writes "his, himself, His"
- N2226 (L20): reads French-language tokens → writes "ét, dé, ère"
- N5082 (L24): reads "dwindle, skyrocket, grows" → writes "rising, rose, increased"

## Implication for the paper

### §2.4 revision (applied)

The method performs desuperposition via subtraction. A single contrast
isolates one axis of variation. Multi-contrast triangulation isolates the
shared axis, averaging out pair-specific signals. The readout quality
depends on how well the contrast targets a single axis.

Observed: single-pair W_U projection recovers 1–11% for compositional
cases, multi-contrast triangulation recovers 49–101% for the same cases,
and entity-identity cases recover 84–97% from single-pair alone.

### MLP finding (new for paper)

MLP neurons implement read→gate→write with zero neuron reuse across 18
tested contrasts. This is observed data, not a model of how MLPs work
in general — we tested 67 out of 10,240 neurons at each layer.

## Quantitative summary

| Case | Single-pair token rec. | Multi-contrast shared rec. | Shared reads as |
|------|----------------------|--------------------------|-----------------|
| IOI (Mary/John) | 84% | — (not needed) | Mary, mary, Maryland |
| Capital (Paris) | 97% | — (not needed) | Paris, Marseille, French |
| Agent swap | 147% | — (not needed) | cat, Catal, kat |
| Caught cold | 1% | 77% | doctor, Doctor, doctors |
| Theft/moral | 4% | 75% | evade, hoped, evasion |
| Hot dog food | 11% | 49% | crispy, charred, cooked |
| Some/all | -40% | 101% (partitive) | some, Some, SOME |
| Metaphor cold | 59% | — (single-pair works) | lowered, below, freezing |
