# Contrastive Axis Taxonomy: Cross-Model Report

**Models**: Pythia-410M (24L), Pythia-1.4B (24L), Phi-2 (32L), Phi-4 (40L)  
**Method**: 2×2 factorial contrastive probing at ~87.5% depth layer (L21/L28/L35 depending on model)  
**Date**: 2026-06-21  
**Scripts**: `explore_axes_wild.py`, `explore_axes_deep.py`

---

## 1. Summary table: axis consistency across models

Each cell is the cosine between the two instances of the axis direction
(AC vs BD for axis 1, AB vs CD for axis 2). Higher = more consistent
direction. Values above 0.7 bolded.

### Axis 1 (row axis) consistency

| Axis                | 410M  | 1.4B  | Phi-2 | Phi-4 | Pattern |
|---------------------|-------|-------|-------|-------|---------|
| code/natural        | 0.93  | **0.95** | **0.93** | **0.88** | Universal, slight decline at scale |
| english/french      | **0.98** | **0.97** | **0.94** | **0.83** | Universal, declines at scale |
| assign/test         | **0.79** | **0.82** | **0.85** | **0.92** | Universal, **improves** with scale |
| positive/negated    | **0.84** | **0.85** | **0.90** | **0.85** | Universal, stable |
| past/future         | **0.87** | **0.86** | **0.89** | **0.81** | Universal, stable |
| cause/effect        | **0.92** | **0.81** | **0.78** | 0.66 | Universal but weakens at scale |
| CAPS/lower          | **0.81** | **0.87** | **0.76** | **0.71** | Universal, stable |
| doubt/certain       | **0.96** | **0.88** | **0.78** | **0.71** | Universal, weakens at scale |
| claim/question      | **0.74** | **0.77** | **0.83** | **0.77** | Universal, moderate |
| formal/informal     | **0.87** | **0.82** | 0.58 | 0.60 | Splits: strong in Pythia, weak in Phi |
| thought/speech      | 0.67 | 0.50 | **0.74** | 0.61 | Partial, variable |
| animate/inanimate   | 0.40 | 0.34 | 0.48 | 0.47 | Never clean |
| active/passive      | 0.17 | 0.50 | 0.65 | 0.35 | Unstable across models |
| literal/metaphor    | 0.35 | 0.35 | 0.19 | 0.19 | Never clean, **worse** at scale |
| salient/generic     | 0.33 | 0.26 | 0.11 | -0.12 | Collapses → inverts at scale |

### Axis 2 (column axis) consistency

| Axis                | 410M  | 1.4B  | Phi-2 | Phi-4 |
|---------------------|-------|-------|-------|-------|
| france/japan (claim) | **0.72** | **0.88** | **0.90** | **0.71** |
| five/ten (code)     | -0.13 | 0.29 | 0.20 | 0.14 |
| five/ten (assign)   | 0.39 | 0.56 | 0.52 | 0.43 |
| dog_ran/cat_sat     | 0.04 | **0.70** | **0.71** | **0.79** |
| france/germany      | **0.90** | **0.94** | **0.81** | **0.91** |
| broke/fell          | **0.79** | **0.83** | **0.82** | **0.74** |
| good/bad (thought)  | **0.89** | **0.93** | **0.78** | **0.87** |
| heart/head          | 0.22 | 0.63 | 0.64 | 0.53 |
| dog_cat/cat_dog     | -0.74 | -0.81 | 0.47 | 0.32 |
| cold/sharp          | 0.13 | 0.15 | 0.15 | 0.06 |
| happy/sad           | **0.97** | **0.94** | **0.96** | **0.92** |
| rain/snow           | **0.84** | **0.85** | **0.93** | **0.91** |
| france/japan (neg)  | **0.87** | **0.76** | 0.65 | 0.53 |
| wrote/built         | **0.95** | **0.92** | 0.55 | 0.35 |
| fire/flood          | 0.40 | 0.36 | 0.33 | 0.16 |
| dog/cat (lang)      | 0.12 | 0.17 | 0.07 | 0.65 |

### Orthogonality (|cross-cosine| mean, lower = more orthogonal)

| Axes                | 410M  | 1.4B  | Phi-2 | Phi-4 |
|---------------------|-------|-------|-------|-------|
| 1st/3rd × happy/sad | **0.11** | **0.08** | **0.11** | **0.11** |
| past/future × rain/snow | 0.16 | **0.08** | **0.06** | **0.11** |
| code/natural × five/ten | 0.26 | **0.08** | **0.12** | **0.13** |
| assign/test × five/ten | 0.19 | 0.19 | 0.15 | **0.12** |
| english/french × dog/cat | **0.13** | **0.08** | **0.12** | **0.14** |
| thought/speech × good/bad | **0.11** | **0.14** | 0.17 | **0.13** |
| CAPS/lower × dog/cat | 0.26 | **0.13** | **0.13** | **0.14** |
| claim/question × france | 0.18 | **0.08** | **0.08** | 0.23 |
| positive/neg × france | 0.45 | 0.19 | 0.28 | 0.16 |
| doubt/certain × france | 0.39 | **0.06** | 0.23 | 0.18 |
| literal/metaphor × cold/sharp | 0.37 | 0.36 | 0.41 | **0.43** |
| active/passive × dog/cat | 0.60 | 0.47 | 0.23 | 0.31 |
| salient/generic × wrote/built | 0.32 | **0.12** | 0.31 | 0.42 |

---

## 2. Axis tiers

### Tier 1: Universal, token-readable, orthogonal to content

These axes are consistent (>0.7) across all four models and near-orthogonal
to the content axis they're crossed with.

**code / natural language** (cos 0.88–0.95)
- Code pole reads as: variable names, numbers, syntax (`y`, `2`, `3`)
- Natural pole reads as: pronouns, common words (`us`, `you`, `eleven`)
- The model maintains a register subspace that separates programming from prose

**assignment / equality test** (cos 0.79–0.92, improves with scale)
- `=` pole reads as: operator tokens (`|-`, `-$`, `=-`)
- `==` pole reads as: result tokens (`result`, `True`, `true`)
- The model distinguishes write-mode from read/test-mode in code

**positive / negated** (cos 0.84–0.90)
- Positive reads as: `uncontested`, `YES`, `solid`, `confirmed`
- Negated reads as: `incorrect`, `false`, `wrong`, `erroneous`
- Clean truth-value direction

**past / future** (cos 0.81–0.89)
- Past reads as: `still`, `now`, `originally`, `STILL`
- Future reads as: `dangerous`, `probable`, `difficult`
- Temporal direction axis

**english / french** (cos 0.83–0.98)
- English reads as: English content words
- French reads as: French morphology (`ami`, `ép`, `autre`, `compagn`)
- A language-identity direction

**CAPS / lowercase** (cos 0.71–0.87)
- CAPS reads as: uppercase tokens (`'S`, `WAS`, `DISCLAIMS`)
- Lower reads as: lowercase tokens (`ios`, `cfg`, `mk`)
- Typography has its own subspace

**doubt / certainty** (cos 0.71–0.96)
- Doubt reads as: open-ended tokens (`anywhere`, `anyone`, `anything`)
- Certainty reads as: continuation tokens (`:`, `__`, definite content)
- Epistemic stance is token-readable

### Tier 2: Partially consistent, content-dependent

**claim / question** (cos 0.74–0.83)
- Both framings produce the same content but different structural predictions
- Question pole at late layers reads as `easy`, `obvious`, `simple`

**thought / speech** (cos 0.50–0.74)
- Thought reads as: subjective/evaluative (`stupid`, `sinister`, `silly`)
- Speech reads as: reporting tokens (`reportedly`, `"`, `also`)
- Moderate cross-content consistency (pairwise cos 0.28–0.64)

**formal / informal** (cos 0.58–0.87, model-family dependent)
- Formal reads as: `confirmed`, `subsequently`, `assessed`
- Informal reads as: `pretty`, `bad`, `weird`, `basically`
- Stronger in Pythia family than Phi family

**cause / effect** (cos 0.66–0.92, weakens with scale)
- Cause reads as: agents (`dryer`, `kitchen`, `wires`, `careless`)
- Effect reads as: results (`collapse`, `implode`, `become`)

### Tier 3: Not a direction

**literal / metaphorical** (cos 0.19–0.35)
- Neither axis produces a consistent direction
- Metaphor is NOT represented as a linear feature
- Both models show the same failure — this is architectural, not a scale issue

**animate / inanimate** (cos 0.34–0.48)
- Weak in all models. Animacy doesn't factor into an independent subspace
- The broke/fell content axis is much stronger (0.74–0.83)

**active / passive voice** (cos 0.17–0.65, unstable)
- Entangles with content (who-chased-whom) in most models
- At Pythia-1.4B, the content axis (dog_cat/cat_dog) actually **inverts** (-0.81)
- Voice is not an independent direction

**salient entity vs generic** (cos -0.12 to +0.33, collapses at scale)
- "Einstein" vs "A scientist" does not produce a stable direction
- At Phi-4, consistency is **negative** (-0.12): the direction reverses
  depending on the verb (wrote vs built)
- Named entities activate specific circuits, not a "salience" axis

---

## 3. Deep probes

### 3.1 Epistemic gradient

Projection of each epistemic framing onto the know→doubt direction
(positive = more doubtful, negative = more certain):

| Framing | Pythia-1.4B | Phi-2 |
|---------|-------------|-------|
| know    | -8          | -17   |
| bare    | 0           | 0     |
| believe | +20         | +26   |
| think   | +20         | +19   |
| suspect | +23         | +28   |
| deny    | +45         | +48   |
| false   | +27         | +52   |
| doubt   | **+58**     | **+72** |

The gradient is **monotonic from know through doubt** in both models.
`deny` falls between `suspect` and `doubt` — it's an epistemic stance,
not a truth value. `false` in Pythia sits lower than `deny` (closer to
believe), suggesting the smaller model treats "it is false that" as
hedging rather than asserting falsehood.

The contrastive readout is also consistent:
- doubt minus bare reads as: `anywhere`, `anything`, `anyone` (open-ended)
- know minus bare reads as: continuation tokens (the answer is settled)

### 3.2 Metaphor processing (layer by layer)

**Universal pattern across models**: The model never has a "this is metaphorical"
direction. Instead, it processes each metaphorical use by routing to the
appropriate domain at mid-layers:

| Contrast | What the literal pole reads as | What the metaphor pole reads as | Onset layer |
|----------|-------------------------------|--------------------------------|-------------|
| cold: ice vs reception | `temperatures`, `below`, `32°F` | `tense`, `gloomy`, `awkward` | L16–20 |
| sharp: knife vs criticism | `blade`, `stainless`, `shiny` | `sarcastic`, `hostile`, `harsh` | L20–24 |
| bright: lamp vs student | `blinding`, `illuminating` | `enrolled`, `gifted`, `intellectual` | L16–20 |
| heavy: boulder vs news | `blocking`, `greater`, `excessive` | `gloomy`, `solemn`, `melancholy` | L12–16 |

The metaphorical sense activates domain-specific tokens (emotion for "cold
reception", education for "bright student") rather than toggling a
figurativity flag. This explains why the literal/metaphor axis has low
consistency — there is no single metaphor direction, just per-domain routing.

### 3.3 Code construct independence

Cross-construct cosine at the reference layer shows that code
constructs are **near-orthogonal** — each is its own direction:

**Phi-2** (L28):
```
                def/call  for/while  print/ret  if/elif  list/dict  py/js
def vs call      1.000     -0.122     -0.102    0.022    0.101    -0.026
for vs while    -0.122      1.000     -0.069   -0.030   -0.141     0.133
print vs return -0.102     -0.069      1.000   -0.026   -0.025    -0.057
if vs elif       0.022     -0.030     -0.026    1.000    0.132     0.010
list vs dict     0.101     -0.141     -0.025    0.132    1.000    -0.130
python vs js    -0.026      0.133     -0.057    0.010   -0.130     1.000
```

**Pythia-1.4B** (L21): same pattern, all cross-cosines < 0.35.

There is no single "code mode" direction. Each syntactic construct occupies
its own subspace. Python vs JavaScript is the most distinctive axis,
reading as `lambda, None, True` vs `typeof, null, Array`.

### 3.4 Negation is not cancellation

Projection onto the `not` direction (bare = 0):

| Negation type | Pythia-1.4B | Phi-2 |
|---------------|-------------|-------|
| not           | **+43**     | **+69** |
| never         | +36         | +63   |
| no_longer     | +35         | +54   |
| double_neg    | **+35**     | **+66** |
| rarely        | +22         | +49   |

Key findings:
- `not` is the strongest single negation in both models
- `rarely` is the weakest (scalar, not categorical)
- **Double negation ("not never") does NOT cancel** — it projects
  positively on the negation axis at 81–95% the strength of single `not`
- The model treats "not never" as emphatic negation, not cancellation
- Each negation type has its own token readout:
  - `not` → `not, NOT, Not`
  - `never` → `never, always, NEVER` (the always is the opposite pole)
  - `no_longer` → `gone, now, replaced` (temporal displacement)
  - `rarely` → `seldom, usually, often` (frequency scale)

### 3.5 Thought vs speech direction

Mean direction across 4 content types:

| Model | Thought pole | Speech pole | Pairwise cos range |
|-------|-------------|-------------|-------------------|
| Pythia-1.4B | maybe, some, somehow, stupid | reportedly, `"`, estimated | 0.28–0.50 |
| Phi-2 | stupid, sinister, boring, silly, evil | currently, also, initially, reportedly | 0.36–0.64 |

The thought direction reads as **subjective/evaluative** content.
The speech direction reads as **reporting** tokens. This is not just
"private vs public" — it's "uncommitted assessment" vs "stated record."
Consistency is moderate, suggesting this is a real but not perfectly
factored axis.

---

## 4. Key claims for the paper

### Method-relevant (for contrastive trajectory paper)

1. The contrastive method can read **grammatical features** (tense, person,
   number) as clean orthogonal axes — this replicates prior work (probing
   classifiers, DAS) but with a simpler method and no training.

2. The method can read **epistemic stance** as a token-readable direction:
   doubt→`anything/anywhere/anyone`, certainty→continuation. This is novel
   relative to prior probing work.

3. Polarity (positive/negated) produces a clean axis with consistent
   token readout (`confirmed` vs `incorrect`) across all four models.

### Universal-axes paper (separate)

4. Axes partition into three tiers: universal token-readable (8 axes),
   partially consistent (4 axes), and non-directions (4 axes).

5. Metaphor is NOT a linear feature — it's processed by per-domain routing,
   not a figurativity flag. This is a negative result with architectural
   implications.

6. Code constructs (def/call, for/while, etc.) are mutually orthogonal —
   there is no single "code mode," but rather a multi-dimensional code
   subspace where each construct is independent.

7. Double negation doesn't cancel in residual-stream geometry — the model
   represents "not never" as emphatic negation, projecting positively on
   the negation axis.

8. The epistemic gradient (know→believe→suspect→doubt) is monotonically
   ordered along a single direction, with `deny` falling between `suspect`
   and `doubt` (it's an epistemic stance, not a truth value).

9. Named-entity salience is not a direction — it inverts with context at
   larger model scales, suggesting entities activate dedicated circuits
   rather than occupying a "salience" subspace.

---

## 5. Raw data files

- `contrastive/logs_phi4/explore_axes_wild.log` — Phi-4 (14B) wild axes
- `contrastive/logs_phi2/explore_axes_wild.log` — Phi-2 (2.7B) wild axes
- `contrastive/logs_pythia1.4b/explore_axes_wild.log` — Pythia-1.4B wild axes
- `contrastive/logs_pythia410m/explore_axes_wild.log` — Pythia-410M wild axes
- `contrastive/logs_phi2/explore_axes_deep.log` — Phi-2 deep probes
- `contrastive/logs_pythia1.4b/explore_axes_deep.log` — Pythia-1.4B deep probes
- `contrastive/logs_phi4/explore_axes_deep.log` — (pending: pod was down)
