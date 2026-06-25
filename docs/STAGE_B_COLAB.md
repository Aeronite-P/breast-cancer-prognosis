# Stage B — Running the Image Classifier on Google Colab

Stage B trains a neural network to classify **benign vs. malignant** breast histopathology
images (BreakHis). It needs a GPU, so it runs in the cloud (free) rather than on your laptop.

## Why Colab (not your Mac)
Training image models is ~10–50× faster on a GPU. Your Mac has no NVIDIA GPU, but Google
Colab gives you one for free. The genomics work (Stages A) stays local; only the image work
moves to Colab.

## Steps
1. Go to <https://colab.research.google.com> (sign in with a Google account).
2. **File → Upload notebook →** choose `notebooks/stage_b_breakhis_colab.ipynb`.
3. **Runtime → Change runtime type → Hardware accelerator = GPU →** Save.
4. **Runtime → Run all.** The first run downloads ~4 GB (a few minutes), then trains.
5. When it finishes, copy the **test AUC, accuracy, and ROC figure** into `LAB_NOTEBOOK.md`
   here in the repo (with the date), following the four-part format.

## What to expect / look for
- The benign-vs-malignant task is relatively easy, so a good run typically reaches a high
  test AUC. **What matters scientifically** is that we used a *patient-level* split (no patient
  in two splits), so the score reflects real generalization, not memorization.
- If the AUC looks suspiciously perfect (≈1.0), double-check the split didn't leak — that's the
  exact failure mode the patient-level grouping is designed to prevent.

## How this connects to the project
Stage B proves the image pipeline works on easy data. **Stage C** then applies the same ideas
(load → features → predict) to the much harder TCGA whole-slide images for the *survival*
question, and **Stage D** fuses those image features with the genes from Stage A.

> Note: the trained Stage B model and the 4 GB download are **not** committed to the repo
> (too large). The notebook + this guide make it fully reproducible.
