# Few-Shot Fine-Tuning Performance Analysis

This document summarizes the cross-database performance of a Q-Former model pre-trained on EvalMI under varying data availability fractions.

## Methodology
- **Pre-training:** EvalMI
- **Target Datasets:** AIGIQA-20K (A20K) and AGIQA-3K (A3K)
- **Fractions:** 0% (Zero-Shot), 5%, 10%, 20% of the training split
- **Architectures Tested:**
  1. `evalmi_baseline_qf.pth`: 1-layer Linear Regressor
  2. `evalmi_baseline_qf_2.pth`: 2-layer MLP Regressor

## Final Experimental Results

| Dataset | Regressor Architecture | Data Fraction | Test SRCC | Test PLCC | Best Epoch |
|---------|------------------------|---------------|-----------|-----------|------------|
| **A20K** | Linear (`qf`) | **0% (Zero-Shot)** | 0.8131 | 0.8062 | - |
| **A20K** | Linear (`qf`) | 5% | 0.8281 | 0.8819 | 10 |
| **A20K** | Linear (`qf`) | 10% | 0.8445 | 0.8885 | 8 |
| **A20K** | Linear (`qf`) | 20% | 0.8680 | 0.9058 | 8 |
| **A20K** | MLP (`qf_2`) | **0% (Zero-Shot)** | 0.8081 | 0.7957 | - |
| **A20K** | MLP (`qf_2`) | 5% | 0.8392 | 0.8822 | 12 |
| **A20K** | MLP (`qf_2`) | 10% | 0.8487 | 0.8904 | 2 |
| **A20K** | MLP (`qf_2`) | 20% | 0.8629 | 0.9004 | 11 |
|---------|------------------------|---------------|-----------|-----------|------------|
| **A3K**  | Linear (`qf`) | **0% (Zero-Shot)** | 0.8057 | 0.8202 | - |
| **A3K**  | Linear (`qf`) | 5% | 0.8511 | 0.8885 | 5 |
| **A3K**  | Linear (`qf`) | 10% | 0.8543 | 0.8964 | 6 |
| **A3K**  | Linear (`qf`) | 20% | 0.8704 | 0.9069 | 8 |
| **A3K**  | MLP (`qf_2`) | **0% (Zero-Shot)** | 0.8261 | 0.8320 | - |
| **A3K**  | MLP (`qf_2`) | 5% | 0.8576 | 0.8993 | 7 |
| **A3K**  | MLP (`qf_2`) | 10% | 0.8691 | 0.9130 | 12 |
| **A3K**  | MLP (`qf_2`) | 20% | **0.8802** | **0.9183** | 4 |

## Key Takeaways & Analysis
1. **Zero-Shot Baseline:** The model performs reasonably well without any target data (~0.81 SRCC on A20K and ~0.80-0.82 SRCC on A3K).
2. **Few-Shot Improvements:** Fine-tuning with just 5% of the data provides a noticeable jump, pushing performance into the 0.83–0.85 SRCC range. Continuing to 20% of the training data pushes SRCC up to **0.86–0.88**.
3. **Linear vs MLP:** The **MLP** regressor generally outperforms the Linear regressor under few-shot conditions, especially on A3K where it hits the highest overall score of `0.8802` SRCC at a 20% data fraction. However, the Linear regressor is slightly more stable in Zero-Shot performance on A20K.

All detailed logs and the comprehensive CSV are located in `results/finetune_all_results.csv`.
