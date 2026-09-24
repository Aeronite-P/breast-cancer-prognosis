# Stage C — Survival from Tumor Slides (Colab guide)

Stage C tests whether a tumor's **physical appearance** (its whole-slide image) predicts
survival — the imaging counterpart to Stage A's genes. It runs on Google Colab with a GPU.

## Why this is the hard stage
TCGA slides are **gigapixel files (~0.5–1 GB each)**. You can't load one whole, and you can't
download them all. The notebook handles this by **streaming one slide at a time**: download →
cut into tiles → keep tissue tiles → turn them into features with a pretrained network →
average into one vector per patient → delete the slide before the next one.

## How to run
1. [colab.research.google.com](https://colab.research.google.com) → **File ▸ Upload notebook** →
   `notebooks/stage_c_tcga_slides_colab.ipynb`
2. **Runtime ▸ Change runtime type ▸ GPU**
3. **Runtime ▸ Run all**

## What to expect (read this)
- **It's slow.** Each slide is a big download plus tiling. The default `N_PATIENTS = 60` may take
  **30–90 minutes**. Start there to confirm the pipeline runs end-to-end.
- **The result is preliminary and noisy.** With a small patient subset and low magnification, the
  C-index will bounce around. That's expected — Stage C is a proof-of-concept, not a final number.
- **WSI pipelines are finicky.** If a cell errors (OpenSlide install, a corrupt slide, GDC
  hiccup), paste the error and we'll fix it. Individual bad slides are skipped automatically.

## Knobs (in Step 3)
- `N_PATIENTS` — raise it (e.g., 150–300) once the small run works, for a stronger result.
- `PATCHES` — tissue tiles sampled per slide (more = better features, slower).
- `DOWNSAMPLE` — magnification level; lower = more detail but much slower.

## After it runs
Record your **C-index** and the **`N_PATIENTS`** you used in `LAB_NOTEBOOK.md`. If the C-index is
clearly above 0.5, images carry survival signal — which sets up **Stage D (fusion)**: combine
these per-patient image vectors with the Stage A gene features and test whether both together
beat either alone.

> The notebook, downloaded slides, and features are **not** committed (slides are huge and
> transient). The notebook + this guide make it reproducible.
