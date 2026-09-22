# Digital Knee X-ray Dataset Profile

Profile generated from the locally downloaded Mendeley dataset on 16 September 2026. The image files remain outside Git under `data/`. The manifest generator is `ml/data/build_manifest.py`.

## Current findings

- The download contains `MedicalExpert-I` and `MedicalExpert-II`.
- Each expert folder contains 1,650 PNG files organised into five KL-label directories.
- The two expert directories represent the same underlying image collection in most cases, not 3,300 independent patients.
- Content-hash grouping is required before any split.
- Eleven content groups have a 0-versus-1 expert-label disagreement and must be excluded from the first supervised baseline or handled in a separately documented adjudication experiment.
- Seventeen duplicate files occur within each expert folder.
- Images have two observed dimensions, 300×162 and 640×161.
- No patient identifier was found in the folder structure or filenames. The first experiment therefore cannot claim patient-separated evaluation.

## First-model policy

1. Run `python ml/data/build_manifest.py` from the repository root.
2. Treat `image_hash` as the grouping key. Never split files directly by path.
3. Use one canonical row per unique image content.
4. Exclude `label_status=disagreement` from the first five-class baseline.
5. Report the absence of patient identifiers as a limitation.
6. Keep the final test hashes locked and never tune against them.

The CC BY 4.0 licence and the original Mendeley record must remain documented with the experiment. Licence permission does not remove the need to describe label quality, provenance, or the limitations of the evaluation design.

