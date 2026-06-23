# Session findings — June 21–23, 2026

## Findings not for the contrastive paper (separate work)

### 1. Heads do double duty: structural constant + content variable

H19@L5 writes a constant "noun-in-determiner-phrase" signal (cos 0.5–0.7 
with any det-noun frame, near-zero for non-noun contexts) with an orthogonal 
content-specific component carrying edibility/properties. Sweep of all 32×32 
heads confirms this is widespread: dual-role heads concentrate in L26–28, 
with 30+ heads showing both structural and content components at >30% energy 
each, roughly orthogonal (|cos|<0.3 between mean and PC1).

**Promising direction**: Systematic taxonomy of head roles via the 
constant/variable decomposition. The constant component is interpretable as 
syntactic function; the variable is content routing. This could be a paper 
about how attention heads multiplex structural and semantic signals.

### 2. Noun identity is frame-invariant; properties are not

Across 6 different syntactic frames ("The hot dog was", "I have a dog", 
"Earlier there were many dogs", etc.), noun-identity heads (H3, H21, H11 
at L28) produce the same contrastive direction with cos 0.86–0.94. But 
property-encoding (hot/cold, tall/short, fast/slow) in the SAME heads 
produces cos 0.02–0.17 across frames.

**Implication**: The model has dedicated noun-identity features that are 
syntactically abstract, but properties are encoded in content-specific 
directions that don't generalize. This is consistent with nouns being 
"entities" and properties being "relations" — different computational 
status.

### 3. Cooking verbs uniquely override noun identity

Systematic test (N=30 concrete nouns × 12 adjective types): only cooking-
method adjectives (fried 29/30, grilled 30/30, roasted 30/30) force food 
predictions on non-food nouns. No other adjective class tested (deadly, 
radioactive, broken, enormous, adorable, etc.) achieves >0/30 food override, 
though several force their own domain predictions (deadly→afraid 30/30, 
frozen→melt 29/30, flying→crashed 30/30).

"Raw" does NOT force the food frame (1/30) — it recognizes existing food 
but doesn't make non-food edible. Linguistic fit: "raw" = unprocessed 
(general), while "fried" = cooked (implies food context).

Abstract nouns resist even cooking verbs: fried 2/10, grilled 4/10.

### 4. Axis→token→override pipeline

Full pipeline for 6 scenarios: extract axis from multi-contrast, read 
through W_U, test token as modifier. Finding: when the contrastive axis 
has high cross-pair consistency (>0.7), the W_U readout surfaces tokens 
that function as frame-forcers. When consistency is <0.5, the direction 
is still causal but not token-nameable.

Examples:
- flying axis (cos 0.74) → reads `future, enabled, technology` → these 
  tokens as modifiers force crash-frame at 87–93%
- frozen axis (cos 0.44) → reads `oop, paradox` (junk) → 7% override
- stolen axis (cos 0.67) → reads `rightfully, threatened` → 93% override

**Promising direction**: Axis consistency as a predictor of whether a 
contrastive direction has token-form that functions as an input feature.

### 5. W_E and W_U are orthogonal in Phi-2

cos(W_U[fried], W_E[fried]) = -0.01. Weights are not tied. The input 
embedding of "fried" and the output detector for "fried" share no 
structure. W_U[fried] is a learned detector for "states that should 
predict fried," not a copy of the word's embedding.

W_U[fried] is fully dense: all 2560 elements nonzero, 50% of energy 
in 320 dimensions, 90% in 1160. The eatability direction's cos=0.06 
alignment with W_U[fried] is built from 2560 individually tiny 
contributions — consistent with superposition.

### 6. Readability predicts causality (r=+0.15)

Across 1774 head contrastive measurements (16 pairs × 4 layers × 32 
heads), outputs reading as real English words through W_U have 57% 
higher causal alignment (|cos| with prediction-site contrastive) than 
outputs reading as subword fragments. The most causally important head 
contributions are almost always token-readable (top-10 all ≥0.8 
readability). But many readable outputs are not causal (readability is 
necessary, not sufficient).

### 7. SVD is the wrong decomposition for contrastive features

The eatability direction is 1D (cos 0.88–0.93 with each contrast). 
SVD of all adj-noun states spreads it across 40 PCs because SVD 
maximizes variance over noun identity (dog/cat/rod), not food-compound 
semantics. The best single SVD PC captures cos=0.33 with eatability. 
Contrastive directions are the model's natural coordinate system; SVD 
is not.

### 8. Potential universal-axes paper

Cross-model axis taxonomy (Pythia-410M/1.4B, Phi-2, Phi-4) with 16 
2×2 factorial designs is written up in contrastive/docs/axes_report.md. 
Key claims:
- 8 universal token-readable axes (code/natural, positive/negated, 
  past/future, english/french, assign/test, CAPS/lower, doubt/certain, 
  claim/question)
- Metaphor is NOT a linear feature — per-domain routing
- Epistemic gradient (know→doubt) is monotonically ordered
- Double negation doesn't cancel in residual geometry
- Code constructs are mutually orthogonal (no single "code mode")

## Scripts created this session

- explore_axes_wild.py — 16 factorial axis experiments
- explore_axes_deep.py — epistemic gradient, metaphor, negation, code
- hotdog_multicontrast.py — triangulation from multiple contrasts
- hotdog_sublayer.py — sub-layer (attn vs MLP) decomposition  
- hotdog_value_decomp.py — per-head V→O at dog position
- hotdog_pca_decomp.py — SVD of noun-position states
- hotdog_head_pc.py — head outputs projected onto PCs
- hotdog_h19_constant.py — H19's structural vs content components
- hotdog_head_read_dir.py — head read directions, linear combinations
- hotdog_routing.py — follow attention routing dog→was
- hotdog_causal_edible.py — causal injection of eatability
- causal_truth_and_dimensions.py — truth causality + SVD critique
- axis_to_token_pipeline.py — full axis→token→override for 6 scenarios
- head_dual_roles.py — sweep all heads for structural/content split
- head_contrastive_sweep.py — heads across diverse prompt structures
- readability_vs_causality.py — token-form predicts causal relevance
- hotdog_axis_decomp.py — semantic axis decomposition (negative result)
