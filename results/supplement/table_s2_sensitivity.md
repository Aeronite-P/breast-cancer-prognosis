# Table S2 — Sensitivity and robustness analyses (n = 1,033)

HR = hazard ratio of the image risk per 1 SD in a Cox model with age, stage, T, N (+ the listed adjustment). ΔC = C-index difference vs clinical alone (paired bootstrap).

| Analysis | Estimate [95% CI] | Test |
|---|---|---|
| Image risk beyond age + stage + T + N (A1, primary) | HR 1.41 [1.22, 1.64] | p = 1.3e-05 |
|   … using site-held-out image risk | HR 1.27 [1.08, 1.49] | p = 0.006 |
|   … adjusted for slide file size (A4) | HR 1.40 | p = 2.3e-05 |
|   … adjusted for tissue composition, k = 8 (B3) | HR 1.31 [1.11, 1.53] | p = 0.0016 |
|   … adjusted for tissue composition, k = 12 (B3) | HR 1.29 [1.09, 1.52] | p = 0.0031 |
|   … without the blank-slide patient (C2c) | HR 1.41 [1.22, 1.64] | p = 1.3e-05 |
|   … image model on 100 tiles per patient (C2a) | HR 1.42 [1.22, 1.65] | p = 1.4e-05 |
|   … image model without non-tissue tiles (C1, cluster 2b) | HR 1.38 [1.19, 1.59] | p = 4.1e-05 |
|   … adjusted for PAM50 subtype (D3) | HR 1.41 [1.21, 1.65] | p = 2.2e-05 |
|   … adjusted for scan magnification (D5) | HR 1.42 [1.22, 1.65] | p = 9.8e-06 |
|   … progression-free interval instead of death (D1) | HR 1.26 [1.08, 1.48] | p = 0.0044 |
|   … disease-specific survival (D1) | HR 1.52 [1.24, 1.85] | p = 8e-05 |
|   … only the 789 not used to design the methods, frozen model (D2; 23 deaths) | HR 1.33 [0.86, 2.05] | p = 0.2 |
|   … same 789, progression-free interval (D2; 58 events) | HR 1.08 [0.83, 1.42] | p = 0.55 |
| Image C-index, random split / unseen hospitals | 0.641 / 0.640 |  |
| Image C-index, 100 tiles per patient (C2a) | 0.634 [0.58, 0.69] |  |
| Genes + images vs images alone (H4): ΔC | +0.019 [-0.023, +0.062] | p = 0.38 |
| Full model, learned weights (A2): ΔC vs clinical | +0.016 [-0.005, +0.036] | p = 0.13 |
| Fixed weights 1:1:1 (C2b): ΔC vs clinical | -0.001 [-0.041, +0.040] | p = 0.96 |
| Fixed weights 2:1:1 (C2b): ΔC vs clinical | +0.013 [-0.015, +0.041] | p = 0.34 |
| Fixed weights 3:1:1 (C2b): ΔC vs clinical | +0.015 [-0.006, +0.037] | p = 0.15 |
| Fixed weights 1:1:0 (C2b): ΔC vs clinical | +0.007 [-0.028, +0.042] | p = 0.69 |
| Fixed weights 2:1:0 (C2b): ΔC vs clinical | +0.016 [-0.006, +0.039] | p = 0.15 |
| Scrambled outcomes, 10 shuffles (genes / clinical) | 0.496 / 0.485 | ≈ 0.5 expected |
| Full model over 5 seeds | 0.723 ± 0.005 |  |
