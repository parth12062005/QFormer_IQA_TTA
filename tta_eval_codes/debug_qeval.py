import os
import pandas as pd
_SPLIT_DIR = "/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/QFormer_IQA_TTA/important split files-20260527T062853Z-3-001/important split files/QEval_new"
CSV_PATHS = [
    os.path.join(_SPLIT_DIR, "qeval_train_full_gen_responses_PT1_normalized.csv"),
    os.path.join(_SPLIT_DIR, "qeval_val_full_gen_responses_PT1_normalized.csv"),
    os.path.join(_SPLIT_DIR, "qeval_test_full_gen_responses_PT1_normalized.csv"),
]
EMBED_OUT_DIR = "/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/dataset/embeddings/qeval"

all_image_names = set()
for csv_path in CSV_PATHS:
    df = pd.read_csv(csv_path)
    all_image_names.update(df["image_name"].tolist())
all_image_names = sorted(all_image_names)

remaining = []
for name in all_image_names:
    flat_name = name.replace("/", "_")
    out_path = os.path.join(EMBED_OUT_DIR, flat_name.replace(".png", ".npz").replace(".jpg", ".npz"))
    if not os.path.exists(out_path):
        remaining.append((name, out_path))

print(f"Remaining: {len(remaining)}")
for name, opath in remaining[:10]:
    print(f"{name} -> {opath}")
