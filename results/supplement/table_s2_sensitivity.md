# Table S2 — Sensitivity and robustness analyses (n = 1,033)

HR = hazard ratio of the image risk per 1 SD in a Cox model with age, stage, T, N (+ the listed adjustment). ΔC = C-index difference vs clinical alone (paired bootstrap).

| Analysis | Estimate [95% CI] | Test |
|---|---|---|
| Image risk beyond age + stage + T + N (A1, primary) | HR 1.40 [1.20, 1.62] | p = 2.5e-05 |
|   … using site-held-out image risk | HR 1.28 [1.08, 1.51] | p = 0.005 |
|   … adjusted for slide file size (A4) | HR 1.38 | p = 4.4e-05 |
|   … adjusted for tissue composition, k = 8 (B3) | HR 1.29 [1.10, 1.52] | p = 0.0025 |
|   … adjusted for tissue composition, k = 12 (B3) | HR 1.28 [1.09, 1.51] | p = 0.0041 |
|   … without the blank-slide patient (C2c) | HR 1.40 [1.20, 1.62] | p = 2.5e-05 |
|   … image model on 100 tiles per patient (C2a) | HR 1.41 [1.21, 1.64] | p = 2.0e-05 |
|   … image model without non-tissue tiles (C1, cluster 2b) | HR 1.37 [1.18, 1.59] | p = 5.5e-05 |
| Image C-index, random split / unseen hospitals | 0.641 / 0.640 |  |
| Image C-index, 100 tiles per patient (C2a) | 0.634 [0.58, 0.69] |  |
| Full model, learned weights (A2): ΔC vs clinical | +0.017 [-0.011, +0.045] | p = 0.21 |
| Fixed weights 1:1:1 (C2b): ΔC vs clinical | +0.003 [-0.037, +0.044] | p = 0.88 |
| Fixed weights 2:1:1 (C2b): ΔC vs clinical | +0.016 [-0.014, +0.045] | p = 0.29 |
| Fixed weights 3:1:1 (C2b): ΔC vs clinical | +0.017 [-0.006, +0.040] | p = 0.15 |
| Fixed weights 1:1:0 (C2b): ΔC vs clinical | +0.013 [-0.024, +0.050] | p = 0.49 |
| Fixed weights 2:1:0 (C2b): ΔC vs clinical | +0.019 [-0.004, +0.044] | p = 0.12 |
| Scrambled outcomes, 10 shuffles (genes / clinical) | 0.496 / 0.490 | ≈ 0.5 expected |
| Full model over 5 seeds | 0.715 ± 0.006 |  |
