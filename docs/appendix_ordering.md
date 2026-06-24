# Appendix: Ordering Mechanism

*Moved from §7 of the main paper. This section uses the contrastive method
on a different task type (transitive reasoning) and is self-contained.*

---

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
between the two contrastive vectors. Separately, PCA of the 6 permutations'
raw hidden states reveals a 2D structure (SVD: 60% + 28%) where orderings
sharing the same bottom entity cluster together; this geometric analysis is
supplementary to the contrastive results.

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
