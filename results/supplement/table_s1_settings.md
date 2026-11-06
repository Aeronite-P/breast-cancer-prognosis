# Table S1 — Model and analysis settings

| Part | Setting | Value |
|---|---|---|
| Slides | Diagnostic H&E slide per patient | smallest if several (full run); GDC download, MD5-verified |
| Tiles | Size / magnification | 224 × 224 px read at level 0 = full scan resolution (scanner 40× for 900 slides, 20× for 119, unknown 14; intended ~20× was not applied — see PREREGISTRATION §11) |
| Tiles | Tissue filter | HSV saturation mean > 25 and value mean < 235 |
| Tiles | Number per patient | up to 200, taken in a fixed random order (seed 0) |
| Features | Model | Phikon (owkin/phikon, ViT), CLS token, 768 numbers per tile |
| Image model | Architecture | gated attention-MIL, hidden 128, dropout 0.4, linear Cox head |
| Image model | Training | Adam, learning rate 1e-3, weight decay 1e-3, 120 epochs, full batch, Cox loss |
| Gene model | Inputs / model | 50 PAM50 genes, standardized; linear Cox head (same trainer) |
| Clinical model | Inputs / model | age, AJCC stage, T, N (median-imputed); Cox with ridge alpha 1.0 |
| Fusion | Late fusion | sum of z-scored risks (equal weights); stacking = Cox on z-scored risks, nested CV |
| Validation | Splits | 5-fold KFold (shuffle, seed 42); GroupKFold by hospital (5 folds) |
| Statistics | Uncertainty | bootstrap 95% CI (1,000); paired bootstrap ΔC (2,000); seed 42 |
| Statistics | Calibration | cross-fitted one-variable Cox recalibration; quintiles; Brier (IPCW) at 3 and 5 yr |
| Software | Key versions | Python 3.9.6, torch 2.8.0, scikit-survival 0.23.1, lifelines 0.30.0 |
