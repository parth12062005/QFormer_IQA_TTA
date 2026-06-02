import os
import pandas as pd

results_dir = "results"
ablation_dirs = [d for d in os.listdir(results_dir) if "ablation_a20k_layer_" in d]

all_summaries = []
for d in ablation_dirs:
    # Extract layer number
    layer_num = int(d.split("layer_")[-1])
    summary_path = os.path.join(results_dir, d, "summary.csv")
    
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        df["unfrozen_layer"] = layer_num
        all_summaries.append(df)

if all_summaries:
    master_df = pd.concat(all_summaries, ignore_index=True)
    master_df = master_df.sort_values("unfrozen_layer")
    
    # Save to a master CSV
    master_df.to_csv("layer_ablation_results.csv", index=False)
    
    # Generate a markdown table
    with open("layer_ablation_results.md", "w") as f:
        f.write("# A20K Single-Layer Fine-Tuning Ablation\n\n")
        f.write("Only the specified layer (plus query tokens and regressor) was unfrozen during a 15-epoch fine-tuning run on 20% of A20K.\n\n")
        f.write("| Unfrozen Layer | Best Epoch | Test SRCC | Test PLCC |\n")
        f.write("|---|---|---|---|\n")
        for _, row in master_df.iterrows():
            f.write(f"| Layer {row['unfrozen_layer']} | {row['best_epoch']} | {row['test_srcc']:.4f} | {row['test_plcc']:.4f} |\n")
    print("Aggregation complete. Results saved to layer_ablation_results.md")
else:
    print("No summary.csv files found to aggregate.")
