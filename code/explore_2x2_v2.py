"""
2x2 factorial v2: better prompt endings and prediction-flipping.

Design principles:
- Endings where the next token is content-determined, not frame-determined
- Prediction flipping via antonyms, swapped agents, different outcomes
- Avoid "The X was" endings that force function words

Usage: .venv/Scripts/python.exe contrastive/code/explore_2x2_v2.py
"""
import sys
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/phi-2"

CASES = [
    # === Animal: what animal is it? ===
    # Content axis: chase scene vs pet description
    # Prediction axis: cat vs dog
    ("animal_identity",
     # A: chase + cat outcome
     "The dog chased the cat through the garden and cornered it near the "
     "fence. The animal that was trapped was the",
     # B: pet description + cat outcome
     "The small furry animal purred loudly and rubbed against her leg, "
     "then curled up on the warm blanket. The pet was a",
     # C: chase + dog outcome (swap agents)
     "The cat chased the dog through the garden and cornered it near the "
     "fence. The animal that was trapped was the",
     # D: pet description + dog outcome
     "The large animal wagged its tail excitedly and fetched the ball "
     "from across the yard. The pet was a"),

    # === Temperature: what does she feel? ===
    # Content axis: water vs air
    # Prediction axis: hot vs cold
    ("hot_cold",
     # A: hot water
     "The water had been boiling for ten minutes. She touched the surface "
     "and it was",
     # B: hot air
     "The desert sun had been beating down all day. She stepped outside "
     "and it was",
     # C: cold water
     "The water had been in the freezer for an hour. She touched the "
     "surface and it was",
     # D: cold air
     "The blizzard had been raging all night. She stepped outside "
     "and it was"),

    # === What did they eat? ===
    # Content axis: breakfast vs dinner context
    # Prediction axis: meat vs vegetable
    ("food",
     # A: breakfast + meat
     "For breakfast he fried some bacon and eggs in a pan and ate the",
     # B: dinner + meat
     "For dinner she roasted a whole chicken with herbs and ate the",
     # C: breakfast + cereal/bread
     "For breakfast he poured some cereal into a bowl with milk and ate the",
     # D: dinner + salad/vegetables
     "For dinner she tossed some lettuce and tomatoes with dressing and ate the"),

    # === What happened to the team? ===
    # Content axis: basketball vs soccer
    # Prediction axis: won vs lost
    ("sports_outcome",
     # A: basketball + won
     "The basketball team scored a three-pointer in the final second to "
     "take the lead. They",
     # B: soccer + won
     "The soccer team scored a goal in stoppage time to take the lead. "
     "They",
     # C: basketball + lost
     "The basketball team missed a free throw in the final second and "
     "fell behind. They",
     # D: soccer + lost
     "The soccer team missed a penalty kick in stoppage time and fell "
     "behind. They"),

    # === What does the doctor say? ===
    # Content axis: cancer vs fracture
    # Prediction axis: serious vs benign
    ("diagnosis",
     # A: cancer signs + serious
     "The scan showed a large tumor that had spread to nearby organs. The "
     "doctor told the patient the diagnosis was",
     # B: fracture + serious
     "The x-ray showed a compound fracture with bone fragments near the "
     "artery. The doctor told the patient the diagnosis was",
     # C: cancer signs + benign
     "The scan showed a small cyst that had not changed in five years. The "
     "doctor told the patient the diagnosis was",
     # D: fracture + benign
     "The x-ray showed a hairline crack that was already healing on its "
     "own. The doctor told the patient the diagnosis was"),

    # === Where did they go? ===
    # Content axis: sick vs hungry
    # Prediction axis: hospital vs restaurant
    ("destination",
     # A: sick + hospital
     "The man collapsed on the sidewalk clutching his chest and could "
     "barely breathe. The ambulance rushed him to the",
     # B: injured + hospital
     "The woman fell off her bicycle and broke her arm in two places. "
     "The ambulance rushed her to the",
     # C: celebration + restaurant
     "The couple had just gotten engaged and wanted to celebrate with "
     "champagne. They drove straight to the",
     # D: hungry + restaurant
     "The family had been driving for hours and the children were "
     "starving. They drove straight to the"),

    # === Guilty vs innocent ===
    # Content axis: murder vs theft
    # Prediction axis: guilty vs not guilty
    ("verdict",
     # A: murder + guilty evidence
     "The defendant's DNA was found on the murder weapon and three "
     "witnesses saw him at the scene. The jury found him",
     # B: theft + guilty evidence
     "The defendant was caught on camera stealing the jewelry and the "
     "items were found in his car. The jury found him",
     # C: murder + innocent evidence
     "The defendant had a confirmed alibi with hotel records placing "
     "him in another city on the night of the murder. The jury found him",
     # D: theft + innocent evidence
     "The defendant had a receipt proving he purchased the jewelry "
     "and the camera footage was proven to be doctored. The jury found him"),

    # === What did they build? ===
    # Content axis: wood vs stone
    # Prediction axis: house vs wall
    ("construction",
     # A: wood + house
     "The carpenter spent months cutting timber and fitting wooden beams "
     "together. He built a",
     # B: stone + house (should also predict house?)
     "The mason spent months cutting limestone blocks and laying them "
     "with mortar. He built a",
     # C: wood + fence
     "The carpenter spent an afternoon nailing wooden planks between "
     "the posts along the property line. He built a",
     # D: stone + wall
     "The mason spent an afternoon stacking flat stones along the edge "
     "of the garden. He built a"),

    # === What color? ===
    # Content axis: sky vs grass
    # Prediction axis: blue/green vs red/brown
    ("color",
     # A: clear sky → blue
     "She looked up at the clear sky on a perfect summer day. The sky "
     "above her was",
     # B: healthy grass → green
     "She looked down at the lawn that had been watered every day all "
     "summer. The grass beneath her feet was",
     # C: sunset sky → red/orange
     "She looked up at the sky just as the sun was setting behind the "
     "mountains. The sky above her was",
     # D: autumn grass → brown
     "She looked down at the lawn after three months without rain in "
     "the heat of late summer. The grass beneath her feet was"),
]


def get_top5(logits, tok):
    probs = torch.softmax(logits, -1)
    vals, idxs = torch.topk(probs, 5)
    return [(tok.decode([int(idxs[i])]).strip(), round(float(vals[i]), 4))
            for i in range(5)]


def get_topk_str(logits, tok, k=5, largest=True):
    vals, idxs = torch.topk(logits, k, largest=largest)
    return ", ".join(tok.decode([int(idxs[j])]).strip()[:12] for j in range(k))


def main():
    print(f"Loading {MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    for p in model.parameters():
        p.requires_grad_(False)

    NL = model.config.num_hidden_layers
    W_U = model.lm_head.weight.detach()

    for label, text_a, text_b, text_c, text_d in CASES:
        texts = {"A": text_a, "B": text_b, "C": text_c, "D": text_d}
        outs = {}
        preds = {}

        for key, text in texts.items():
            ids = tok(text, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                out = model(torch.tensor([ids], device=DEV),
                            output_hidden_states=True)
            outs[key] = out
            top5 = get_top5(out.logits[0, -1], tok)
            preds[key] = top5

        a1 = preds["A"][0][0]
        b1 = preds["B"][0][0]
        c1 = preds["C"][0][0]
        d1 = preds["D"][0][0]

        ab = "=" if a1 == b1 else "≠"
        cd = "=" if c1 == d1 else "≠"
        ac = "=" if a1 == c1 else "≠"
        valid = (a1 == b1) and (c1 == d1) and (a1 != c1)

        print(f"\n{'='*120}")
        print(f"  {label}  {'✓ VALID 2x2' if valid else '✗ invalid'}")
        print(f"  A{ab}B: {a1},{b1}   C{cd}D: {c1},{d1}   A{ac}C")
        for key in "ABCD":
            print(f"  {key}: \"{texts[key]}\"")
            print(f"     → {preds[key]}")

        # Only show full trajectories for valid or near-valid cases
        if not valid:
            for key in "ABCD":
                del outs[key]
            torch.cuda.empty_cache()
            continue

        pairs = [
            ("AB", "same pred, diff content"),
            ("CD", "same pred, diff content"),
            ("AC", "same content, diff pred"),
            ("BD", "same content, diff pred"),
        ]

        for pair, desc in pairs:
            k1, k2 = pair[0], pair[1]
            print(f"\n  --- {k1}-{k2} ({desc}) ---")
            print(f"  {k1}: \"{texts[k1]}\"")
            print(f"  {k2}: \"{texts[k2]}\"")
            print(f"  {'L':>3} | {'norm':>6} | "
                  f"{'→ ' + k1 + ' pole':^42} | {'→ ' + k2 + ' pole':^42}")
            print(f"  {'-'*100}")

            for L in range(8, NL + 1, 4):
                h_x = outs[k1].hidden_states[L][0, -1, :]
                h_y = outs[k2].hidden_states[L][0, -1, :]
                dh = h_x - h_y
                norm = float(dh.norm() / h_x.norm())
                logits_d = dh @ W_U.T
                pos = get_topk_str(logits_d, tok, 5, largest=True)
                neg = get_topk_str(logits_d, tok, 5, largest=False)
                print(f"  {L:>3} | {norm:.3f} | {pos:>42} | {neg:>42}")

        for key in "ABCD":
            del outs[key]
        torch.cuda.empty_cache()

    print(f"\n{'='*120}")
    print("DONE")


if __name__ == "__main__":
    main()
