# A20K Single-Layer Fine-Tuning Ablation

Only the specified layer (plus query tokens and regressor) was unfrozen during a 15-epoch fine-tuning run on 20% of A20K.

| Unfrozen Layer | Best Epoch | Test SRCC | Test PLCC |
|---|---|---|---|
| Layer 0 | 13 | 0.8501 | 0.8990 |
| Layer 1 | 14 | 0.8522 | 0.8978 |
| Layer 2 | 14 | 0.8650 | 0.9080 |
| Layer 3 | 4 | 0.8679 | 0.9074 |
| Layer 4 | 15 | 0.8679 | 0.9094 |
| Layer 5 | 4 | 0.8687 | 0.9088 |
| Layer 6 | 11 | 0.8680 | 0.9093 |
| Layer 7 | 13 | 0.8639 | 0.9089 |
| Layer 8 | 15 | 0.8686 | 0.9104 |
| Layer 9 | 4 | 0.8699 | 0.9117 |
| Layer 10 | 4 | 0.8668 | 0.9076 |
| Layer 11 | 7 | 0.8614 | 0.9045 |
