import os
import csv
import random

random.seed(42)

base_dir = "/media/parth/021f75bf-bae8-49ef-86a5-28ca19171835/parth/EvalMi-50K"

# 1. Read prompts
with open(os.path.join(base_dir, "prompts.txt"), "r") as f:
    prompts = [line.strip() for line in f.readlines()]

# 2. Read imgnames
with open(os.path.join(base_dir, "imgnames.txt"), "r") as f:
    imgnames = [line.strip() for line in f.readlines()]

# 3. Read mos1
with open(os.path.join(base_dir, "mos1.txt"), "r") as f:
    mos1 = [float(line.strip()) for line in f.readlines()]

# Create the data
rows = []
for i in range(len(imgnames)):
    img_name = imgnames[i]
    m1 = mos1[i]
    
    # Extract index from name
    try:
        idx_str = os.path.splitext(os.path.basename(img_name))[0]
        idx = int(idx_str) - 1
        idx = min(max(0, idx), len(prompts) - 1)
        prompt = prompts[idx]
    except Exception as e:
        prompt = "A photo"
        
    rows.append({
        "image_name": img_name,
        "prompt": prompt,
        "gen_answer": prompt,
        "gt_score": m1 # Using mos1 exclusively as requested
    })

# Shuffle and split
random.shuffle(rows)
total = len(rows)
train_split = int(0.7 * total)
val_split = int(0.1 * total) + train_split

train_data = rows[:train_split]
val_data = rows[train_split:val_split]
test_data = rows[val_split:]

def write_csv(filename, data):
    with open(os.path.join(base_dir, filename), "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "prompt", "gen_answer", "gt_score"])
        writer.writeheader()
        for row in data:
            writer.writerow(row)

write_csv("evalmi_train.csv", train_data)
write_csv("evalmi_val.csv", val_data)
write_csv("evalmi_test.csv", test_data)

print(f"Re-created CSVs using ONLY mos1. {len(train_data)} train, {len(val_data)} val, and {len(test_data)} test rows.")
