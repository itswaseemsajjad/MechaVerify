# Synthetic-to-real test set (planned)

This is a small, expert-labeled set of hand-drawn and textbook mechanism diagrams.
It measures transfer from rendered schematics to real drawings, and it is the one
part of MechaVerify whose images the generator does not render. Every label must
stay exact. Draw-from-spec items get theirs by construction, and textbook and
hand-drawn diagrams are labeled by hand from the depicted mechanism, never scraped.

## Target

- About 60 to 100 items. The set is small by design, so results are indicative.
- Both task families and all seven tasks, spread across difficulty.
- Split id `real`, which the verifier and the evaluation harness support.

## Sources

Record the provenance and license of every image.

- Textbook figures from *Design of Machinery* (Norton), *Theory of Machines and
  Mechanisms* (Uicker, Pennock and Shigley) or *Mechanisms and Dynamics of
  Machinery* (Mabie and Reinholtz). Use figures you have the right to use, or redraw
  them.
- Hand-drawn sketches of gear trains, four-bars, linkage skeletons and pocket
  sections, photographed or scanned. These change the rendering completely and raise
  no licensing questions.
- Public-domain or openly licensed figures, such as Wikimedia Commons diagrams, USPTO
  patent drawings and open courseware.
- Leave out anything whose answer you cannot verify independently from the drawing.

## Draw-from-spec items

`draw-worklist` writes a list of mechanisms to draw by hand (`worklist.md`) together
with their known labels (`labels.jsonl`), so these items need no manual labeling.

```bash
python -m mechaverify.cli draw-worklist --out real_drawn --per-task 6
# draw each item in real_drawn/worklist.md, photograph or scan it,
# and save it as real_drawn/photos/<id>.png
python -m mechaverify.cli build-drawn --images real_drawn/photos \
    --labels real_drawn/labels.jsonl --out real_drawn/manifest.jsonl
```

## Textbook and hand-drawn diagrams

1. Put the images in a folder, for example `real/images/`.
2. Write the labeling template.
   ```bash
   python -m mechaverify.cli real-template --out real/labels.csv
   ```
3. Replace the example row with one row per image. The required columns are
   `image`, `family`, `task`, `question`, `answer_value` and `answer_kind`. Optional
   columns are `choices` (options separated by `|`), `tolerance` (relative, for
   numeric answers), `difficulty`, `source` and `symbolic_description`.
   `mechaverify/realset.py` documents each one. Reuse the synthetic question wording
   where possible so the prompts match the benchmark.
4. Compute each answer from the mechanism. Count teeth and meshes for sense and
   ratio, order the link lengths for Grashof, count links and joints for mobility,
   and read the corner radius, width, depth and tool diameter for DFMA. Have a second
   person check the labels.
5. Build the manifest. This also checks that every image exists.
   ```bash
   python -m mechaverify.cli build-real --images real/images --labels real/labels.csv \
       --out real/manifest.jsonl
   ```

## Evaluate

Both manifests run through the same harness and verifier as the synthetic splits.

```bash
python -m mechaverify.cli evaluate --manifest real/manifest.jsonl \
    --provider gemini --model gemini-2.5-pro --conditions D --out runs/real_eval.json
python -m mechaverify.cli evaluate --manifest real_drawn/manifest.jsonl \
    --provider gemini --model gemini-2.5-pro --conditions D --out runs/real_drawn_eval.json
```

The main measurement is condition D, the drawing itself. Condition S is optional.
Draw-from-spec items already carry a symbolic description. For CSV items, fill
`symbolic_description` only if you transcribe the mechanism to text.

## Reporting

Report the set size, the sources and the labeling procedure, including who labeled
and how agreement was checked.
