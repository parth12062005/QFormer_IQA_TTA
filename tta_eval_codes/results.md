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

## Parameter Change Analysis (20% Fine-Tuning)

To understand which parts of the network drive the few-shot adaptation, we trained both checkpoints on 20% of the data and calculated the relative average absolute change (`avg(abs(trained - orig)) / avg(abs(orig))`) for every layer. 

The results reveal a powerful and consistent pattern across **all combinations** of datasets and architectures. The parameters that undergo the most dramatic changes are heavily concentrated in the **bias terms of the cross-attention query and value projections**, particularly in the early-to-mid layers of the Q-Former.

### Top 5 Most Changed Layers (Relative)

**AIGIQA-20K (Linear Regressor)**
1. `encoder.layer.6.crossattention.self.query.bias` (11.80% change)
2. `encoder.layer.0.crossattention.self.query.bias` (11.26% change)
3. `embeddings.position_embeddings.weight` (10.80% change)
4. `encoder.layer.8.crossattention.self.query.bias` (8.38% change)
5. `encoder.layer.10.crossattention.self.key.weight` (8.15% change)

**AIGIQA-20K (MLP Regressor)**
1. `encoder.layer.0.crossattention.self.query.bias` (15.74% change)
2. `encoder.layer.6.crossattention.self.query.bias` (14.81% change)
3. `encoder.layer.2.crossattention.self.query.bias` (11.97% change)
4. `encoder.layer.0.crossattention.self.value.bias` (11.71% change)
5. `encoder.layer.4.crossattention.self.query.bias` (11.12% change)

**AGIQA-3K (Linear Regressor)**
1. `encoder.layer.0.crossattention.self.query.bias` (8.19% change)
2. `encoder.layer.0.crossattention.self.value.bias` (6.57% change)
3. `encoder.layer.6.crossattention.self.query.bias` (6.56% change)
4. `encoder.layer.2.crossattention.self.query.bias` (6.26% change)
5. `encoder.layer.4.crossattention.self.query.bias` (5.36% change)

**AGIQA-3K (MLP Regressor)**
1. `encoder.layer.0.crossattention.self.query.bias` (8.11% change)
2. `encoder.layer.8.crossattention.self.value.bias` (7.88% change)
3. `encoder.layer.4.crossattention.self.value.bias` (7.66% change)
4. `encoder.layer.0.crossattention.self.value.bias` (7.66% change)
5. `encoder.layer.6.crossattention.self.query.bias` (7.08% change)

> [!TIP]
> **Implications for TTA:** This provides massive insight for Test-Time Adaptation! Instead of updating the `LayerNorms` or the raw `query_tokens`, the model naturally prefers adapting the **cross-attention query/value biases** (especially in layers 0, 2, 4, 6) when bridging domain gaps. Targeting these specific bias parameters during TTA could yield dramatic stability and efficiency improvements compared to full network tuning.
